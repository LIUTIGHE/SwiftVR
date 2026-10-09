# M8/A1：历史 GT 记录与 P1 任务后训练

基于 fork `feature/training-forward` 的 `39114d3e384cb8c04021c03e8bb66bc0289cb70b` 及用户回传日志/视觉观察整理。代码默认值不等于历史服务器 run_config；没有配置或视觉结论的地方不补造。本文不宣称新训练已经完成。

## 1. 历史：GT 与输出梯度都不是第一次

|阶段|核对的实现/记录|支持的结论与边界|
|---|---|---|
|早期 Stage-3|`swiftvr/training/stage3.py`：RGB L1 + GT 相邻帧差分 MSE。服务器 `stage3_recon_pilot_100` 日志记录 adapter-only、0 ReAE +23.873M DiT 参数，step100 GT PSNR 27.0849|GT 输出监督已经实施；日志 PASS 是执行状态，不能替代当时的视觉失败判断。不能把后来 Stage-3 2000步记录缩写成全阶段只训练100步。|
|早期 Tiny Conditional Decoder|`tools/train_tiny_decoder_formal_ddp.py`、`swiftvr/training/tiny_decoder.py`：固定 cached z_SR，仅更新 decoder；GT/Original rendering 双目标像素和LPIPS|GT+LPIPS 已存在，但不更新上游DiT。历史空间 p2/p8 伪影不可与现在四帧时间弱相位混称。|
|B2B D768|`swiftvr/training/b2b_joint.py`：z_LQ-v -> compact decoder -> RGB，latent不detach；默认0.05 velocity NMSE +0.05 cosine +1 teacher RGB L1 +0.5 GT RGB L1，无LPIPS/GAN/temporal|GT 已穿过紧凑decoder更新DiT；历史逐run实际权重仍以run_config为准。|
|D768 staged/joint|`tools/train_b2b_aggressive_compare_ddp.py`：staged 将decoder LR设为0但保留反传。保存的历史memo报告staged500：Teacher/GT PSNR 28.846/25.400；joint500：28.922/25.847|视觉仍“既不像GT忠实，也不像Teacher锐利”，staged较多HF噪声。冻结decoder/回传DiT不能作为本轮新机制。两个指标不能与当前不同decoder的绝对值直接比较。|
|M8-C|`swiftvr/training/m8_joint.py`，V1/V2 Stage-A-centric；V2把decoder与Transformer梯度路径分开|负结果保留，不归因于唯一latent drift，也不等同于本轮GT感知目标。|
|M8-A/FullBoost|当前以TA/Stage-A velocity监督恢复压缩网络；FullBoost候选TA1500未升级正式milestone|量化有小幅变化，但用户视觉认为丢失细节没有被明显救回。不能把全部增益单独归因于UltraVideo。|
|A1 E99|同latent模仿Original Decoder，已有L2/LPIPS/temporal|不等于实际A1对真实HR的误差已充分用于训练M8。|
|R2、GAN|R2曾设计/短试或撤销；视频对抗仍是计划|不能说LPIPS恢复或GAN已经有正式失败/成功结论。|

历史文件：Library `Pasted text(7).txt`（Stage3日志）、`Pasted markdown(3).md`（D768 staged/joint记录）、2026-09-22项目交接memo。最新B1视觉由用户在本对话提供：C24/C48同帧出现四帧细节退化；Original较轻、以两帧明暗交替为主。外层chunk长度不是首要干预点，但共同时间展开/状态机制仍未排除。

## 2. 本轮问题与固定项

**问题不是“GT loss能否下降”，而是GT感知后训练能否避免旧式回归的细节折中。**

- 正式保留：M8-A30k + A1 E99，约500.8 GFLOPs/output frame（1920x1088、MAC=2 FLOPs口径）。
- 新run初始化默认TA FullBoost1500，只是候选，不覆盖正式节点。
- 固定Original Encoder、A1 E99权重；更新完整M8 D1024/H8/L20 MoE，shared1024 +12 routed256、top2不变。
- 同一目标：`prediction = A1(z_LQ - M8(z_LQ))`，`clamp=False`；RGB梯度穿过A1。
- 控制P1-R：GT L1 + GT帧差MSE +router；主实验P1-P再加GT LPIPS。不存在teacher RGB/velocity终点约束。
- TA只提供初始化；不需要新GT latent cache；已有teacher cache仅提供历史视图metadata，不读velocity tensor。
- 新意不是GT、冻结decoder或temporal loss本身，而是当前成熟压缩组合的GT感知梯度路径。LPIPS也不保证所有指标提高。
- 原先50:50 sampler原样复用，是全局采样epoch比例，不宣称每个batch都精确50:50。两组相同sampler/seed。
- 13帧128 LR ->384 HR。使用同一个可微A1，但这是whole-clip训练，不宣称已经匹配长视频FIRST/MIDDLE/LAST状态。C24部署不变。

P1-R不是旧B2B双teacher/GT配方的逐项复现，而是当前系统上的回归型对照。建议先给控制1500更新、主实验5000更新，共同比较500/1000/1500；两者scheduler horizon都10000。需要严格同预算时把两条MAX_STEPS设成相同。500/1500不是通用收敛标准；320k presentations（5000*64）也不是独立视频数。

## 3. 新文件与兼容边界

仅新增P1文件，不覆盖`inference.py`/`inference_custom_components.py`或历史trainer，因此不会丢失服务器已有`--stream`改动。

- `swiftvr/training/m8_task.py`：包裹canonical MoE forward、冻结A1输入反传、三类GT目标、离线完整LPIPS加载。
- `swiftvr/data/task_pairs.py`：沿用deterministic legacy views/UltraLR读法、加入HR和严格row身份。
- `tools/materialize_ultravideo_hr_targets.py`：对现有LR manifest补存canonical HR；每clip bundle，分片/同计划resume，LR不改。
- `tools/train_m8_task_posttrain.py`：DDP、FP32主权重/BF16 forward、累积、缓存清理、val13、每次custom hook、可续训state。
- `tools/visual_check_m8_task.sh`：明确的Original预生成/current推理/原尺寸四宫格拼接。
- `tools/run_m8_task_p1.sh`：两条明确配方的启动命令。
- `tests/test_m8_task_posttrain.py`：CPU接口/梯度/目标测试；真实模型须服务器smoke。

Resume限相同数据、代码hash、world size/batch、loss、scheduler与输出run。只可延长max_steps；保存optimizer、rank RNG和实际消费cursor。所有checkpoint新增目录；保存时复制FP32 state，不量化活跃训练权重。

## 4. 数据：先4个clip，再全量

已有完整LR merged manifest：
`outputs/data_audit/ultravideo_lr_train_v1/merged/materialized_views.jsonl`

```bash
python -m unittest discover -s tests -p 'test_m8_task_posttrain.py'
python tools/materialize_ultravideo_hr_targets.py \
  --lr-manifest outputs/data_audit/ultravideo_lr_train_v1/merged/materialized_views.jsonl \
  --output-dir outputs/data_audit/ultravideo_hr_smoke4_v1 \
  --max-clips 4 --decode-batch-size 4 --qa-views 4 --progress-every 1
```

QA左LQ bicubic/右GT：不改flip或时间索引。前几个view中心帧会使用原seed/退化重建LR，与已有LR逐像素比较，失败直接报错。不重新调severity。

HR定义是原canonical crop在3x BOX降采样之前的RGB，8K源依既有规则形成canonical4K crop；不是把HQ上采样。复用原crop函数，传scale-one HR坐标，其末端同尺寸resize是identity。完整20088*13*384*384*3 uint8约107.6 GiB（未计小量metadata），先检查学校磁盘。

四片可以分别在四个终端运行，不需要GPU：
```bash
for S in 0 1 2 3; do
  python tools/materialize_ultravideo_hr_targets.py \
    --lr-manifest outputs/data_audit/ultravideo_lr_train_v1/merged/materialized_views.jsonl \
    --output-dir outputs/data_audit/ultravideo_hr_train_v1/shard_$S \
    --shard-count 4 --shard-index "$S" --decode-batch-size 4
 done
```
此处for循环是顺序执行；并行需要分别执行对应S，不隐式后台启动。中断后同计划加`--resume`；不复用smoke目录作正式full目录。正式trainer要求四片、完整索引覆盖；不需要额外merge。

## 5. LPIPS离线完整权重

只执行一次。使用学校已有AlexNet缓存+lpips包内校准权重；缺失直接报错，绝不下载或随机替代。
```bash
python - <<'PY'
from pathlib import Path
from unittest.mock import patch
import torch, lpips
path=Path('outputs/local_weights/lpips_alex_complete.pt')
if path.exists(): raise FileExistsError(path)
with patch('torch.hub.download_url_to_file', side_effect=RuntimeError('Missing local AlexNet weights; downloads disabled')):
    model=lpips.LPIPS(net='alex', verbose=False).eval()
path.parent.mkdir(parents=True,exist_ok=True)
torch.save(model.state_dict(),path)
print(path)
PY
```
不能直接传lpips包内的`alex.pth`小文件；它通常只有线性校准层。trainer先无下载构建网络，再`strict=True`加载完整state。

## 6. 真实模型smoke（不是质量gate）

```bash
CUDA_VISIBLE_DEVICES=7 python tools/train_m8_task_posttrain.py \
 --base-checkpoint checkpoints_prompt_free_no_time \
 --student-init outputs/b2b/m8a_fullboost_ta_5k_v1/checkpoints/step_00001500 \
 --decoder-checkpoint outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder \
 --legacy-view-cache outputs/b2a/cache_stage_a200k_train_full \
 --val-view-cache outputs/b2a/cache_stage_a200k_val13 \
 --ultra-pairs outputs/data_audit/ultravideo_hr_smoke4_v1/paired_views.jsonl \
 --lpips-weights outputs/local_weights/lpips_alex_complete.pt \
 --output-dir outputs/b2b/p1_gt_smoke2_v1 --path-root . \
 --batch-size 1 --gradient-accumulation-steps 1 --num-workers 0 \
 --smoke-steps 2 --log-every 1
```
smoke专门只对GT loss反传（不靠router loss制造非零梯度），检查M8梯度非零、E/A1梯度为空，并保存完整val13连续帧PNG/指标。没有GPU真实运行前不能标成集成PASS。随后可用四卡torchrun相同参数在新的smoke目录确认DDP。

## 7. 每次custom validation

先在训练之外准备Original，不让训练进程再加载一个大teacher：
```bash
GPU=7 CUSTOM_REF_ROOT=outputs/custom/p1_reference_v1 \
  bash tools/visual_check_m8_task.sh prepare
```
默认两条完整输入是用户指定Wetland1/Mangrove1及其AISR_pred。不再只取前97帧。Original为`checkpoints`完整conditional SwiftVR，绝非Stage-A。

每个val step（包括step0）只运行当前checkpoint，A1不变；LQ|Original / Current|TinyCNN原尺寸960x960 ROI四宫格，FFV1无损RGB8 MKV+连续相位PNG。不要让播放器fit-window缩放后的图代替100%像素检查；可在prepare时设置ROI1/ROI2（SR坐标），随后冻结。

时间对齐复用`compose_4way_video._auto_basic_index`：以实际Current推理帧数N为准，TinyCNN较长M时做规范化长度采样，少于N报错。LQ/Original只截多余尾帧，不改前缀。保存每个TinyCNN源索引。这个历史展示协议不是PTS严格对齐，也不是训练loss的输入。

reference.json绑定输入/TinyCNN/Original checkpoint，校验文件stat避免错误复用；Original PNG保留。Current完整PNG仅作为本次推理临时产物，**无损视频及ROI PNG成功写完才删除**（KEEP_CUSTOM_PNG=1可保留）。视频失败则保留full PNG并报错；使用同一个checkpoint和输出目录执行 `bash tools/visual_check_m8_task.sh compose <checkpoint> <output>` 可只重做拼接，不重跑GPU。reference bank不得手工替换内容。

## 8. 正式配方

```bash
export LPIPS_WEIGHTS=outputs/local_weights/lpips_alex_complete.pt
export HR_ROOT=outputs/data_audit/ultravideo_hr_train_v1
export CUSTOM_REF_ROOT=outputs/custom/p1_reference_v1
export CUDA_VISIBLE_DEVICES=4,5,6,7

RECIPE=reconstruction MAX_STEPS=1500 RUN_DIR=outputs/b2b/p1_gt_reconstruction_v1 \
 bash tools/run_m8_task_p1.sh

RECIPE=perceptual MAX_STEPS=5000 RUN_DIR=outputs/b2b/p1_gt_perceptual_v1 \
 bash tools/run_m8_task_p1.sh
```
顺序跑，不自动抢同一组卡。默认local2*accum8*4=global64；若local2 OOM，改local1/accum16。视图不在线重抽，GT不在线解码原始4K/8K，DataLoader workers/prefetch、非阻塞H2D、A1/LPIPS checkpointing、按log步做host同步；吞吐日志记录loader wait与更新率，不把功耗当成功指标。

新增LPIPS权重0.1只是首个明确配方，不宣称最优。控制也加载LPIPS但只在validation计算，确保比较口径一致。若需要续训，保持同一RECIPE/RUN_DIR等配置，增大MAX_STEPS并在launcher后传`--resume <该run/checkpoints/step_...>`，不改变scheduler10000。

## 9. 结果与停止规则

- val13：GT PSNR/SSIM/MAE，LPIPS，temporal，raw输出越界比例，逐相位MSE；全部13 sample×13帧可视化。
- 每次正式val：两条完整custom的原尺寸四宫格+相位标签；Original/TinyCNN固定。
- PSNR改善但纹理更平、好帧磨平或亮度闪烁更坏：质量失败，不升级milestone。
- 感知主实验改善真实细节且时间/内容可靠，再进入视频GAN；P1纹理仍保守不要求一直等到完美。
- 只在有证据时进一步解冻A1。冻结A1不保证四帧弱相位能被上游完全补偿。
- 这批代码不实现GAN、特征KD、新teacher或推理改造。

工作协议：只写用户fork/本分支；PR4 open/draft/unmerged；不改upstream、不擅自merge。服务器是GPU与视觉事实来源。容器测试、代码提交、服务器smoke、长训完成、视觉通过分别报告。新文件手动同步，不要求服务器git pull/reset。

## 10. 本批验证状态

隔离CPU容器实际运行16项单测：15通过、1跳过。通过项包括替身网络的GT梯度路径、checkpointing前后预测/梯度一致、LPIPS完整state严格加载接口、HR row身份、CLI help、shell语法，以及合成帧源的实际FFV1四宫格编码/解码像素与长度映射。跳过项是原仓库canonical HR裁剪函数的集成测试：隔离目录没有该原文件，服务器完整仓库应执行它。

没有在本容器运行真实LPIPS预训练网络、完整ReAE/M8/A1 checkpoint、CUDA或DDP训练；保存/恢复逻辑已实现，但其完整模型断点续训仍需服务器验证。没有新的质量结果，不应把接口单测或smoke当作视觉PASS。
