我来帮你梳理 RVM 的使用经验。RVM 的官方文档对 `downsample_ratio` 和输入维度有明确建议，我先确认一下当前的官方推荐值，再结合你的 VR 场景给出实践原则。下面按你列的几个点逐一说，结合 VR 场景给一些实践建议。

## 参考链接

- 官方仓库：https://github.com/PeterL1n/RobustVideoMatting
- 推理文档（最关键，downsample_ratio 的官方说明都在这）：https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md
- WACV 2022 论文：https://openaccess.thecvf.com/content/WACV2022/papers/Lin_Robust_High-Resolution_Video_Matting_With_Temporal_Guidance_WACV_2022_paper.pdf

## downsample_ratio 与输入维度

核心机制：模型内部分两阶段，stage 1 把输入降采样后跑编码-解码网络拿到粗 alpha，stage 2 再用 Fast Guided Filter 在原分辨率上做边缘精修。所以 downsample_ratio 要设到「降采样后的分辨率落在 256~512 之间」。

官方给的锚点：HD（1080p）用 0.25，4K 用 0.125。比如 1920×1080 配 0.25 得到 480×270，正好在 256~512 区间。`downsample_ratio=None` 时会自动让降采样后的最长边约为 512px。

对你的 equirectangular ~3000×3000：如果整帧直接喂进去，512/3000 ≈ 0.17，取 0.15~0.18。但更要紧的一条原则是——downsample_ratio 要看画面内容，特写人像用低值就够，全身/人占比小的镜头用高值；而且不是越高越好。equirectangular 里人通常只占画面一小块，等于「人占比很小」的情形，这时候如果用整帧 3000×3000 去算，人脸区域在降采样后只剩几十个像素，抠出来必然糊。所以对 VR 我更推荐先裁人物区域（或先做 equirect→perspective 投影，你之前做过 v360 那套），在透视空间里 matting，再投影回去——这样人物在 ratio 计算里占满画面，效果会好一个量级。

输入维度本身没有硬性整除要求（全卷积），但为了 stage 2 的 guided filter 对齐，建议宽高都保持偶数、最好能被 4 整除，避免上采样错位出现 1px 抖动边。

## 使用 RVM 的几条原则

最容易踩的坑是它是 RNN，不是逐帧独立模型。必须按时间顺序处理，并把 4 个 recurrent state 循环喂回去（rec = [None]*4，然后 fgr, pha, *rec = model(src, *rec, ratio)）。具体说：

- 绝对不要打乱帧顺序、不要在镜头中途重置 state（除非真的切镜头了），否则时序记忆断掉，alpha 会抖。
- 整段镜头内 downsample_ratio 保持不变，中途改会让 state 语义错乱。
- seq-chunk 只影响吞吐（一次塞几帧并行），不影响时序逻辑，双 2080 上可以调大点压满显存。
- 大多数情况用 mobilenetv3 就够，resnet50 提升很小但慢不少。
- fp32 在 2080 上没问题；2080 是 Turing，其实支持 fp16，想提速可以试 fp16，质量差异基本看不出来。

## 去除人物无关的杂点

RVM 对动态背景或背景里有「像人」的东西时，会冒出孤立的 false-positive 小块，这是它没有语义约束导致的。几种处理，从轻到重：

阈值 + 形态学：对 pha 做二值化后 morphological open（先腐蚀再膨胀）能去掉零碎噪点，但会啃边缘，建议只用很小的核（3×3）。

连通域过滤：这是最稳的纯后处理——对 alpha 做连通域分析，按面积排序，只保留最大的若干块（或面积大于阈值的块），把孤立小块清零。对「画面里只有一两个人」的场景几乎一劳永逸。

语义门控（最推荐你用）：你本来就在跑 SAM 3/3.1 做人物分割，那就让 SAM 出一张人物区域的 mask 当 gate，`alpha_final = alpha_rvm * dilate(sam_mask)`。SAM 负责「人在哪」，RVM 负责「头发丝/边缘的软过渡」，两者互补：SAM 边缘硬、RVM 边缘软但会乱跑。这个组合能同时解决杂点和边缘质量。注意 SAM mask 要适当 dilate 几个像素，别把 RVM 的发丝软边裁掉。

## VR 左右眼抠图不统一

这是 VR 特有的硬骨头。问题根源：左右眼有视差（parallax），如果你对 L、R 两路各跑一次 RVM，它们的 recurrent state 是独立演化的，加上视差本身的内容差异，两眼的 alpha 边缘会错位、闪烁不同步。戴上头显后双眼看到不一致的边缘会产生视网膜竞争（retinal rivalry），非常难受，比单眼抠图差还难受。

几个思路：

把左右眼拼成一帧一次性处理。SBS 或 TB 拼成单张大图，对整张跑一次 RVM。这样权重、downsample、时序节拍完全一致，是改动最小、收益明显的做法。缺点是 RVM 不会主动强制两个区域间的几何一致，但「同一次前向、同一套 state 节拍」已经能消掉大部分跨眼闪烁。

单眼 matting + 视差 warp。只对一只眼（比如左眼）抠图，用立体视差/光流把 alpha warp 到另一只眼。优点是天然一致，缺点是 warp 在遮挡边界（disocclusion）会破，发丝区域尤其难，需要对 warp 后的边缘再做一次局部精修或 inpaint。

语义基底统一 + 各眼精修。延续上面 SAM 门控的思路：用 SAM 在两眼分别出 mask，但通过视差把其中一只眼的 mask 对齐校验另一只，保证「人物轮廓」这个低频信息两眼一致；高频边缘各眼用 RVM 各自精修。因为基底一致，残留的边缘差异在头显里不易察觉。

无论哪种，有几条必须守住：两眼用完全相同的 downsample_ratio 和预处理；如果分两路跑，两路的 state 各自独立、绝不混用，但要逐帧同步推进；最后输出前对两眼 alpha 做一次一致性检查（比如统计两眼前景面积差，超阈值就触发 warp 校正）。

如果可以的话，我个人会推荐你走「先 equirect→perspective 投影 → 在透视空间做 SAM+RVM → SBS 拼图统一处理 → 投影回 equirect」这条链路，既解决了等距投影畸变让 RVM 失准的问题，又顺带把两眼放进同一前向里。

