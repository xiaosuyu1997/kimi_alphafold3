# AlphaFold 3: SI Section 3 (Model Architecture) to Code Mapping

This document maps the algorithms and architectural components described in Supplementary Information Section 3 (Model Architecture) of the AlphaFold 3 paper to their corresponding implementations in the [AlphaFold 3 GitHub repository](https://github.com/google-deepmind/alphafold3).

## Table of Contents

1. [Input Embeddings (Section 3.1)](#31-input-embeddings)
2. [Sequence-local Atom Attention (Section 3.2)](#32-sequence-local-atom-attention)
3. [MSA Module (Section 3.3)](#33-msa-module)
4. [Triangle Updates (Section 3.4)](#34-triangle-updates)
5. [Template Embedding (Section 3.5)](#35-template-embedding)
6. [Pairformer Stack (Section 3.6)](#36-pairformer-stack)
7. [Diffusion Module (Section 3.7)](#37-diffusion-module)

---

## 3.1 Input Embeddings

### 3.1.1 Input Embedder

**SI Algorithm 2: InputFeatureEmbedder**

| Component | Code Location | Description |
|-----------|---------------|-------------|
| `InputFeatureEmbedder` | [`alphafold3/model/network/featurization.py`](alphafold3_repo/src/alphafold3/model/network/featurization.py) | Creates initial single (1D) embeddings from input features |
| `create_target_feat` | [`featurization.py`](alphafold3_repo/src/alphafold3/model/network/featurization.py) ~L1-L100 | Constructs target feature vector from token features |
| Atom conditioning | [`atom_cross_attention.py`](alphafold3_repo/src/alphafold3/model/network/atom_cross_attention.py) | Per-atom conditioning in input embedder |

**Key Code References:**
- [`evoformer.py:148-171`](alphafold3_repo/src/alphafold3/model/network/evoformer.py#L148-L171) - `create_target_feat_embedding` function
- [`atom_cross_attention.py:117-352`](alphafold3_repo/src/alphafold3/model/network/atom_cross_attention.py#L117-L352) - `atom_cross_att_encoder`

### 3.1.2 Relative Position Encoding

**SI Algorithm 20: RelativePositionEncoding**

| Component | Code Location | Line Range |
|-----------|---------------|------------|
| `RelativePositionEncoding` | [`featurization.py`](alphafold3_repo/src/alphafold3/model/network/featurization.py) | ~L100-L200 |
| `create_relative_encoding` | [`featurization.py`](alphafold3_repo/src/alphafold3/model/network/featurization.py) | L100-L200 |

**Key Code References:**
- [`evoformer.py:77-91`](alphafold3_repo/src/alphafold3/model/network/evoformer.py#L77-L91) - `_relative_encoding` method

---

## 3.2 Sequence-local Atom Attention

**SI Algorithm 4: AtomAttentionEncoder**
**SI Algorithm 5: AtomAttentionDecoder**
**SI Algorithm 6: AtomTransformer**

| Algorithm | Code Location | Key Functions |
|-----------|---------------|---------------|
| `AtomAttentionEncoder` | [`atom_cross_attention.py:117-352`](alphafold3_repo/src/alphafold3/model/network/atom_cross_attention.py#L117-L352) | `atom_cross_att_encoder` |
| `AtomAttentionDecoder` | [`atom_cross_attention.py:361-420`](alphafold3_repo/src/alphafold3/model/network/atom_cross_attention.py#L361-L420) | `atom_cross_att_decoder` |
| `AtomTransformer` | [`diffusion_transformer.py:331-403`](alphafold3_repo/src/alphafold3/model/network/diffusion_transformer.py#L331-L403) | `CrossAttTransformer` |

**Implementation Details:**
- Per-atom conditioning: [`atom_cross_attention.py:34-96`](alphafold3_repo/src/alphafold3/model/network/atom_cross_attention.py#L34-L96)
- Cross-attention mechanism: [`diffusion_transformer.py:263-329`](alphafold3_repo/src/alphafold3/model/network/diffusion_transformer.py#L263-L329)

---

## 3.3 MSA Module

**SI Algorithm 7: MsaModule**
**SI Algorithm 9: OuterProductMean**
**SI Algorithm 10: MSAPairWeightedAveraging**

| Algorithm | Code Location | Key Classes/Functions |
|-----------|---------------|----------------------|
| `MsaModule` | [`modules.py:535-627`](alphafold3_repo/src/alphafold3/model/network/modules.py#L535-L627) | `EvoformerIteration` |
| `OuterProductMean` | [`modules.py:334-413`](alphafold3_repo/src/alphafold3/model/network/modules.py#L334-L413) | `OuterProductMean` class |
| `MSAPairWeightedAveraging` | [`modules.py:87-129`](alphafold3_repo/src/alphafold3/model/network/modules.py#L87-L129) | `MSAAttention` |

**Key Implementation Details:**
- MSA stack: [`evoformer.py:223-241`](alphafold3_repo/src/alphafold3/model/network/evoformer.py#L223-L241)
- MSA feature creation: [`featurization.py`](alphafold3_repo/src/alphafold3/model/network/featurization.py)

---

## 3.4 Triangle Updates

**SI Algorithm 11: TriangleMultiplicationOutgoing**
**SI Algorithm 12: TriangleMultiplicationIncoming**
**SI Algorithm 13: TriangleAttentionStartingNode**
**SI Algorithm 14: TriangleAttentionEndingNode**

| Algorithm | Code Location | Class Name |
|-----------|---------------|------------|
| `TriangleMultiplicationOutgoing` | [`modules.py:245-332`](alphafold3_repo/src/alphafold3/model/network/modules.py#L245-L332) | `TriangleMultiplication` (equation='ikc,jkc->ijc') |
| `TriangleMultiplicationIncoming` | [`modules.py:245-332`](alphafold3_repo/src/alphafold3/model/network/modules.py#L245-L332) | `TriangleMultiplication` (equation='kjc,kic->ijc') |
| `TriangleAttentionStartingNode` | [`modules.py:131-243`](alphafold3_repo/src/alphafold3/model/network/modules.py#L131-L243) | `GridSelfAttention` (transpose=False) |
| `TriangleAttentionEndingNode` | [`modules.py:131-243`](alphafold3_repo/src/alphafold3/model/network/modules.py#L131-L243) | `GridSelfAttention` (transpose=True) |

---

## 3.5 Template Embedding

**SI Algorithm 17: TemplateEmbedder**

| Component | Code Location | Key Classes |
|-----------|---------------|-------------|
| `TemplateEmbedder` | [`template_modules.py`](alphafold3_repo/src/alphafold3/model/network/template_modules.py) | `TemplateEmbedding` |
| Distogram features | [`template_modules.py`](alphafold3_repo/src/alphafold3/model/network/template_modules.py) | `dgram_from_positions` |

**Integration with Evoformer:**
- Template embedding: [`evoformer.py:170-198`](alphafold3_repo/src/alphafold3/model/network/evoformer.py#L170-L198)

---

## 3.6 Pairformer Stack

**SI Algorithm 18: PairformerStack**

| Component | Code Location | Description |
|-----------|---------------|-------------|
| `PairformerStack` | [`evoformer.py:310-331`](alphafold3_repo/src/alphafold3/model/network/evoformer.py#L310-L331) | Main Pairformer stack |
| `PairFormerIteration` | [`modules.py:415-533`](alphafold3_repo/src/alphafold3/model/network/modules.py#L415-L533) | Single Pairformer iteration |

**Key Components in PairFormerIteration:**
- Triangle Multiplication (Outgoing): Line 468-472
- Triangle Multiplication (Incoming): Line 474-478
- Grid Self-Attention (non-transposed): Line 480-485
- Grid Self-Attention (transposed): Line 487-492
- Transition Block: Line 494-504
- Single representation attention: Line 506-530

---

## 3.7 Diffusion Module

**SI Algorithm 19: SampleDiffusion**
**SI Algorithm 21: DiffusionModule**
**SI Algorithm 22: DiffusionConditioning**
**SI Algorithm 23: FourierEmbedding**
**SI Algorithm 24: DiffusionTransformer**
**SI Algorithm 25: AttentionPairBias**

| Algorithm | Code Location | Key Functions/Classes |
|-----------|---------------|----------------------|
| `SampleDiffusion` | [`diffusion_head.py:297-369`](alphafold3_repo/src/alphafold3/model/network/diffusion_head.py#L297-L369) | `sample` function |
| `DiffusionModule` | [`diffusion_head.py:101-295`](alphafold3_repo/src/alphafold3/model/network/diffusion_head.py#L101-L295) | `DiffusionHead` class |
| `DiffusionConditioning` | [`diffusion_head.py:133-201`](alphafold3_repo/src/alphafold3/model/network/diffusion_head.py#L133-L201) | `_conditioning` method |
| `FourierEmbedding` | [`noise_level_embeddings.py`](alphafold3_repo/src/alphafold3/model/network/noise_level_embeddings.py) | `noise_embeddings` |
| `DiffusionTransformer` | [`diffusion_transformer.py:180-255`](alphafold3_repo/src/alphafold3/model/network/diffusion_transformer.py#L180-L255) | `Transformer` class |
| `AttentionPairBias` | [`diffusion_transformer.py:120-178`](alphafold3_repo/src/alphafold3/model/network/diffusion_transformer.py#L120-L178) | `self_attention` function |

**Additional Diffusion Components:**
- `adaptive_layernorm`: [`diffusion_transformer.py:23-52`](alphafold3_repo/src/alphafold3/model/network/diffusion_transformer.py#L23-L52)
- `adaptive_zero_init`: [`diffusion_transformer.py:54-76`](alphafold3_repo/src/alphafold3/model/network/diffusion_transformer.py#L54-L76)
- `transition_block`: [`diffusion_transformer.py:78-112`](alphafold3_repo/src/alphafold3/model/network/diffusion_transformer.py#L78-L112)

---

## 4. Confidence Heads (Section 4.3)

**SI Algorithm 30: ConfidenceHead**

| Component | Code Location | Description |
|-----------|---------------|-------------|
| `ConfidenceHead` | [`confidence_head.py:29-321`](alphafold3_repo/src/alphafold3/model/network/confidence_head.py#L29-L321) | Main confidence head class |
| pLDDT prediction | [`confidence_head.py:237-251`](alphafold3_repo/src/alphafold3/model/network/confidence_head.py#L237-L251) | Predicted LDDT |
| PAE prediction | [`confidence_head.py:194-236`](alphafold3_repo/src/alphafold3/model/network/confidence_head.py#L194-L236) | Predicted Aligned Error |
| PDE prediction | [`confidence_head.py:160-193`](alphafold3_repo/src/alphafold3/model/network/confidence_head.py#L160-L193) | Predicted Distance Error |
| Experimentally resolved | [`confidence_head.py:252-263`](alphafold3_repo/src/alphafold3/model/network/confidence_head.py#L252-L263) | Experimentally resolved prediction |

---

## 5. High-Level Model Structure

### Main Model Class

| Component | Code Location | Description |
|-----------|---------------|-------------|
| `Model` | [`model.py:216-345`](alphafold3_repo/src/alphafold3/model/model.py#L216-L345) | Main model class |
| `Evoformer` (trunk) | [`evoformer.py:30-348`](alphafold3_repo/src/alphafold3/model/network/evoformer.py#L30-L348) | Evoformer trunk network |
| `DiffusionHead` | [`diffusion_head.py:101-295`](alphafold3_repo/src/alphafold3/model/network/diffusion_head.py#L101-L295) | Diffusion head for structure prediction |

### Main Inference Loop

**SI Algorithm 1: MainInferenceLoop**

| Step | Code Location | Description |
|------|---------------|-------------|
| Recycling loop | [`model.py:280-311`](alphafold3_repo/src/alphafold3/model/model.py#L280-L311) | `recycle_body` function |
| Embedding module | [`evoformer.py:243-347`](alphafold3_repo/src/alphafold3/model/network/evoformer.py#L243-L347) | Evoformer forward pass |
| Diffusion sampling | [`diffusion_head.py:297-369`](alphafold3_repo/src/alphafold3/model/network/diffusion_head.py#L297-L369) | `sample` function |

---

## Quick Reference: File Organization

### Network Modules

| File | Description | Key Components |
|------|-------------|----------------|
| [`evoformer.py`](alphafold3_repo/src/alphafold3/model/network/evoformer.py) | Evoformer trunk | `Evoformer`, `_seq_pair_embedding`, `_relative_encoding` |
| [`modules.py`](alphafold3_repo/src/alphafold3/model/network/modules.py) | Core architectural blocks | `PairFormerIteration`, `EvoformerIteration`, `TriangleMultiplication`, `GridSelfAttention`, `TransitionBlock` |
| [`diffusion_head.py`](alphafold3_repo/src/alphafold3/model/network/diffusion_head.py) | Diffusion module | `DiffusionHead`, `sample`, `_conditioning` |
| [`diffusion_transformer.py`](alphafold3_repo/src/alphafold3/model/network/diffusion_transformer.py) | Diffusion transformer | `Transformer`, `self_attention`, `CrossAttTransformer` |
| [`atom_cross_attention.py`](alphafold3_repo/src/alphafold3/model/network/atom_cross_attention.py) | Atom-level attention | `atom_cross_att_encoder`, `atom_cross_att_decoder` |
| [`confidence_head.py`](alphafold3_repo/src/alphafold3/model/network/confidence_head.py) | Confidence prediction | `ConfidenceHead` |
| [`template_modules.py`](alphafold3_repo/src/alphafold3/model/network/template_modules.py) | Template embedding | `TemplateEmbedding` |
| [`distogram_head.py`](alphafold3_repo/src/alphafold3/model/network/distogram_head.py) | Distogram prediction | `DistogramHead` |
| [`featurization.py`](alphafold3_repo/src/alphafold3/model/network/featurization.py) | Input featurization | `create_target_feat`, `create_msa_feat`, `create_relative_encoding` |
| [`noise_level_embeddings.py`](alphafold3_repo/src/alphafold3/model/network/noise_level_embeddings.py) | Noise level embeddings | `noise_embeddings` |

### Main Model Files

| File | Description |
|------|-------------|
| [`model.py`](alphafold3_repo/src/alphafold3/model/model.py) | Main Model class, `get_predicted_structure`, `create_target_feat_embedding` |
| [`model_config.py`](alphafold3_repo/src/alphafold3/model/model_config.py) | Model configuration classes |
| [`features.py`](alphafold3_repo/src/alphafold3/model/features.py) | Feature definitions |
| [`feat_batch.py`](alphafold3_repo/src/alphafold3/model/feat_batch.py) | Batch data structures |

---

## Notes on Implementation Mapping

### Differences Between SI Algorithms and Code

1. **Algorithm 7 (MsaModule)**: In the code, this is implemented as `EvoformerIteration` in [`modules.py`](alphafold3_repo/src/alphafold3/model/network/modules.py), which handles both MSA and pair representation updates in a single iteration.

2. **Algorithm 18 (PairformerStack)**: Implemented as `PairFormerIteration` in [`modules.py:415-533`](alphafold3_repo/src/alphafold3/model/network/modules.py#L415-L533). The stack is created using `hk.experimental.layer_stack` in [`evoformer.py:310-331`](alphafold3_repo/src/alphafold3/model/network/evoformer.py#L310-L331).

3. **Algorithm 24 (DiffusionTransformer)**: Implemented as `Transformer` class in [`diffusion_transformer.py:180-255`](alphafold3_repo/src/alphafold3/model/network/diffusion_transformer.py#L180-L255).

### Configuration Classes

Most modules have corresponding configuration classes defined using `base_config.BaseConfig`. For example:
- `Evoformer.Config`, `PairFormerIteration.Config`, `DiffusionHead.Config`

These configs are typically defined as nested classes within the main module class and specify hyperparameters like:
- Number of layers/blocks
- Channel dimensions
- Number of attention heads
- Other architectural hyperparameters

### Key Design Patterns

1. **Haiku Modules**: All components inherit from `hk.Module`
2. **Layer Stacks**: Repeated blocks use `hk.experimental.layer_stack`
3. **BFloat16 Context**: Mixed precision training via `utils.bfloat16_context()`
4. **Config-based**: All hyperparameters passed through config objects

---

*Document generated: March 28, 2026*
*Based on AlphaFold 3 commit: `608edb684db9f6fd0e677fea01c4cefc60f8a8aa`*
