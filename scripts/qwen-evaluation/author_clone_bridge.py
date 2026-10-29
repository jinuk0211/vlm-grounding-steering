"""Call pinned author functions with explicit Gemma input/output compatibility.

No local baseline mathematics are used for RUDDER or GrAInS. Imports of CLI
helpers are isolated by AST to avoid executing unrelated dataset/model loaders.
Every isolated function retains its exact author source and code filename.
"""
import ast
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
sys.path.insert(0, str(ROOT / '.author-deps'))
AUTHORS = ROOT / 'upstream-author-20261026'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def author_function(path, name, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def load_authors():
    rudder = load_file('ljw_author_rudder', AUTHORS / 'RUDDER/Qwen/qwen_methods.py')
    gradient_dir = AUTHORS / 'GrAInS/src/attribution/gradient'
    package = types.ModuleType('ljw_author_gradient')
    package.__path__ = [str(gradient_dir)]
    sys.modules[package.__name__] = package
    ig_utils = load_file('ljw_author_gradient.utils', gradient_dir / 'utils.py')
    # The author's Captum integrand/20 quadrature nodes stay unchanged. Only
    # scheduling is bounded to one quadrature sample to fit the shared GPU.
    original_ig = ig_utils.IntegratedGradients
    class BoundedIG(original_ig):
        def attribute(self, inputs, *args, **kwargs):
            kwargs.setdefault('internal_batch_size', inputs.shape[0])
            return super().attribute(inputs, *args, **kwargs)
    ig_utils.IntegratedGradients = BoundedIG
    grad = load_file('ljw_author_gradient.vlm_grad', gradient_dir / 'vlm_grad.py')
    steering = load_file('ljw_author_steering', AUTHORS / 'GrAInS/src/utils/steering.py')
    namespace = {'torch': torch, 'np': np, 'log_softmax': F.log_softmax}
    model_helpers = AUTHORS / 'GrAInS/src/utils/model.py'
    for name in ('get_all_layer_hidden_states', 'get_log_prob', 'get_top_token_indices'):
        author_function(model_helpers, name, namespace)
    namespace['vlm_grad'] = grad
    ablation = author_function(AUTHORS / 'GrAInS/src/token_ablation_vlm.py',
                               'run_token_ablation_contrastive', namespace)
    generation_mask = author_function(AUTHORS / 'RUDDER/Qwen/qwen_pope.py',
                                      'build_generation_mask', {'torch': torch, 'Dict': Dict})
    return types.SimpleNamespace(rudder=rudder, grad=grad, ig_utils=ig_utils,
                                 steering=steering, ablation=ablation, generation_mask=generation_mask)


class NativeBatchProcessor:
    """Let unchanged CARD extraction receive the already native Gemma batch."""
    def __init__(self, batch):
        self.batch = batch
    def __call__(self, **kwargs):
        return dict(self.batch)


@contextlib.contextmanager
def rudder_context(adapter, batch, task, authors, diagnostics=None):
    m = authors.rudder
    layer, alpha = (1, 8.0) if task == 'pope' else (25, 5.0)
    target = m.get_qwenvl_self_attn(adapter.model, layer)
    with torch.no_grad():
        card = m.compute_card_vector_batch(adapter.model, NativeBatchProcessor(batch),
                                           images=[None], questions=['native Gemma prompt'],
                                           layer_idx=layer, pooling='attn', local=False)
    mask = authors.generation_mask(batch) if task == 'pope' else batch['attention_mask'].unsqueeze(-1).float()
    m.mask_carrier.set(mask)
    hook = m.BayesianGatingHookMaskedDynamic(max_alpha=alpha, sensitivity=5.0,
                                            concentration=1.0, carrier=m.mask_carrier,
                                            clamp=(0.05, 1.0), rms_match=False, record=True)
    hook.set_vector(card)
    counts = {'calls': 0, 'active_positions': 0, 'positions': 0, 'max_gate': 0.0}
    def observed(module, inputs, output):
        result = hook(module, inputs, output)
        stats = hook.fetch_last_stats()
        counts['calls'] += 1
        counts['active_positions'] += int((stats['gate'] > 0).sum())
        counts['positions'] += stats['gate'].numel()
        counts['max_gate'] = max(counts['max_gate'], float(stats['gate'].max()))
        if not torch.isfinite(result[0]).all():
            raise ValueError('Nonfinite author RUDDER output')
        return result
    handle = target.register_forward_hook(observed)
    try:
        yield
    finally:
        handle.remove()
        m.mask_carrier.clear()
        if diagnostics is not None:
            diagnostics.update(counts, card_norm=float(card.float().norm()), layer=layer, alpha=alpha,
                               hook_source=hook.__call__.__code__.co_filename)


class NamedLayerView:
    """Map the exact author hook's supported path to Gemma's decoder block."""
    def __init__(self, module, layer):
        self.module, self.layer = module, layer
    def named_modules(self):
        return iter([(f'language_model.model.layers.{self.layer}', self.module)])


class GemmaAttributionAPI:
    """Supply Gemma's token-identity PLE for original token-embedding IG.

    Gemma cannot reverse interpolated embeddings into token IDs. Its documented
    inputs_embeds interface instead accepts precomputed per_layer_inputs. Token
    identities stay fixed along IG; the model still differentiates its ordinary
    context projection of the interpolated embeddings. Image placeholders use
    the same PAD substitution as native Gemma's input_ids forward. This is an
    explicit architecture adapter, not a claim of full-path multimodal IG.
    """
    def __init__(self, model):
        self.native=model
        self.identities=[]
        self.forward_index=0
    def __getattr__(self,name):
        return getattr(self.native,name)
    def eval(self):
        self.native.eval()
        return self
    def get_input_embeddings(self):
        self.identities=[]
        self.forward_index=0
        def embed(ids):
            masks=self.native.model.get_placeholder_mask(ids,None)
            clean=torch.where(masks[0] | masks[1] | masks[2],
                              self.native.config.text_config.pad_token_id,ids)
            identity=self.native.model.language_model.get_per_layer_inputs(clean,None).detach()
            self.identities.append(identity)
            return self.native.get_input_embeddings()(clean)
        return embed
    def __call__(self,*args,**kwargs):
        if kwargs.get('inputs_embeds') is not None:
            assert len(self.identities)==2, 'Expected original positive/negative IG embedding pair'
            identity=self.identities[self.forward_index % 2]
            self.forward_index+=1
            embeds=kwargs['inputs_embeds']
            assert identity.shape[1]==embeds.shape[1]
            kwargs['per_layer_inputs']=identity.expand(embeds.shape[0],-1,-1,-1)
            kwargs['use_cache']=False
        return self.native(*args,**kwargs)


@contextlib.contextmanager
def grains_context(adapter, vector, authors, diagnostics=None):
    layer, alpha = 27, 10.0
    module = adapter.layers[layer]
    counts = {'calls': 0, 'changed_elements': 0, 'total_elements': 0,
              'max_abs_delta': 0.0, 'layer': layer, 'alpha': alpha}
    # Exact tuple contract required by the original hook. Hooks run in insertion
    # order: Tensor -> tuple -> author hook -> Tensor. The author dim=1 norm and
    # 10x multiplier are intentionally retained, including decode-time quirks.
    def wrap(_m, _args, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError('Expected native Gemma Tensor decoder output')
        counts['_before'] = output.detach().clone()
        return (output,)
    def unwrap(_m, _args, output):
        value = output[0]
        before = counts.pop('_before')
        difference = (value.float() - before.float()).abs()
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite original GrAInS steering output')
        counts['calls'] += 1
        counts['changed_elements'] += int((difference > 0).sum())
        counts['total_elements'] += difference.numel()
        counts['max_abs_delta'] = max(counts['max_abs_delta'], float(difference.max()))
        return value
    first = module.register_forward_hook(wrap)
    middle = authors.steering.add_steering_hook(NamedLayerView(module, layer), layer,
                                               vector, alpha=alpha)
    if len(middle) != 1:
        first.remove()
        raise RuntimeError('Author GrAInS hook must attach to exactly one layer')
    last = module.register_forward_hook(unwrap)
    try:
        yield
    finally:
        for handle in [last, *middle, first]:
            handle.remove()
        counts.pop('_before', None)
        if diagnostics is not None:
            diagnostics.update(counts, hook_source=authors.steering.add_steering_hook.__code__.co_filename)


def extract_one(adapter, record, authors):
    from PIL import Image
    with Image.open(record['image']) as source:
        image = source.convert('RGB').resize((448, 448), Image.Resampling.BICUBIC)
    prompt = adapter.processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'text', 'text': record['prompt']}, {'type': 'image'}]}],
        tokenize=False, add_generation_prompt=True)
    values = authors.ablation(GemmaAttributionAPI(adapter.model), adapter.processor, image, prompt,
                              record['chosen'], record['rejected'],
                              top_k=50, top='pos', attribution_method='integrated_gradients')
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite original GrAInS extraction {name}')
    return values


def verify_clone_sources():
    import hashlib
    import subprocess
    info = {}
    for name, commit in [('RUDDER', 'bd149d8dbde6a9097dd02aea80f83d0bd8876761'),
                         ('GrAInS', '8ad78db3040fbb7b0aec9e544a3dcc7b9c050b5b')]:
        repo = AUTHORS / name
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == commit
        assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
        files = subprocess.check_output(['git', '-C', str(repo), 'ls-files'], text=True).splitlines()
        info[name] = {'commit': commit, 'files': {f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in files}}
    return info
