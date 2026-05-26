你在用 MatAnyone，遇到边缘锯齿、效率低、效果反而不如 RVM。这是个比较新的模型，我先查一下它的官方用法和输入分辨率处理机制，确认锯齿是哪个环节造成的。先放参考链接，再说你遇到的三个问题（锯齿、效率低、不如 RVM）其实是同一个设计差异的不同表现。

## 参考链接

- MatAnyone 2 仓库（你用的这个，CVPR 2026 Highlight）：https://github.com/pq-yang/MatAnyone2
- MatAnyone v1 仓库（CVPR 2025）：https://github.com/pq-yang/MatAnyone
- MatAnyone 2 论文：https://arxiv.org/pdf/2512.11782
- v1 论文（讲清楚 memory propagation 和边界 loss 的设计）：https://arxiv.org/html/2501.14677v1

## 先理解它和 RVM 是完全不同的两类模型

这点没搞清楚，参数怎么调都别扭。

最大的区别：MatAnyone 不是 auxiliary-free 的。它每次推理需要一段视频 + 首帧分割 mask 作为输入，mask 一般用 SAM/SAM2 交互式分割得到。也就是说它是「mask 引导 + 记忆传播」的模型，RVM 是「只看 RGB、靠 RNN 记忆」的模型。MatAnyone 的卖点是 target assignment（指定抠哪个人）和在长视频、复杂/模糊背景下的稳定跟踪，靠的是一个 memory bank 做时序传播。

所以「执行效率低」是设计决定的，不是你用错了。RVM 是为实时设计的轻量 RNN，MatAnyone 是 memory-based 的离线模型，本来就重。如果你的素材背景不复杂、也不需要在多人里锁定某一个人，RVM 已经够用，那 MatAnyone 的这套重机制对你就是纯开销——「更大的模型 ≠ 更好的结果」，要看素材是不是它的目标场景。它真正赢 RVM 的地方是：背景里有别的人/类人物体、长视频要稳定锁定同一目标、RVM 会乱跳的那些场景。

## 锯齿的根因：512 这个输入分辨率

这是你问题的核心。RVM 内部有两阶段——低分辨率算粗 alpha、再用 Fast Guided Filter 在原分辨率上精修边缘（上一轮说过）。**MatAnyone 没有这个独立的高分辨率精修阶段**，它输出的 alpha 分辨率基本等于处理分辨率。默认它甚至不限制分辨率，只有当你传 --max_size 且 min(w,h) 超过时才降采样。

所以你把输入压到 512，拿到的就是 512 的 alpha，再 upscale 回 equirect 的 ~3000px，边缘必然变成阶梯状锯齿——这不是模型抠不好，是分辨率被你自己卡死了。RVM 在低分辨率下边缘还行，正是因为它有 guided filter 在原图分辨率上补救，MatAnyone 没有这层兜底，对输入分辨率比 RVM 敏感得多。

另一个叠加因素：它的边界区域质量来自 region-adaptive fusion 和专门的边界 loss，但首帧 mask 的边界质量会被一路传播下去。官方排查建议就是：先看 alpha 输出，发丝和软边应该是平滑渐变而不是硬切；如果边缘不行，回 SAM2 把首帧 mask 修干净再跑一遍。如果你给的 SAM mask 边缘本身是锯齿/粗糙的，结果也会是锯齿。

## 使用原则

几条要守住的：

首帧 mask 决定上限。garbage in → garbage out。花时间在 SAM2 里把首帧 mask 做干净、边界贴合，比调任何推理参数都管用。

别压分辨率，尤其别压到 512。MatAnyone 输出 alpha = 处理分辨率，没有高分辨率精修兜底。--max_size 的存在意义是 4K 素材爆显存时降采样保命（比如设 --max_size 1080），是个妥协手段，不是越小越好。要锐利边缘就得给够分辨率。

用 warmup 稳定首帧。MatAnyone 有用首帧反复「预热」memory bank 的机制（v1 是 n_warmup，默认重复若干帧），跑之前 `--help` 确认一下 v2 的对应参数，开了之后开头几帧的跳变会少很多。

输出 upscale 用 bilinear/bicubic，别用 nearest，否则锯齿是你自己引入的。

## 给你 VR 场景的具体做法

你的痛点（3000×3000 equirect、双 2080、要锐利边缘、要效率）其实和上一轮 RVM 的解法是同一条路，而且对 MatAnyone 更关键，因为它没有低分辨率兜底：

不要整帧 3000×3000 喂模型。先用 SAM 拿到人物 bbox，把人物区域裁出来（或先做 equirect→perspective 投影），在裁切区域的**接近原生分辨率**上跑 matting，再贴/投影回 equirect。这样一举两得：人物在画面里占满，边界像素充足，锯齿消失；同时处理区域变小，速度也上来了，正好缓解 MatAnyone 慢的问题。

如果你的素材本来就是 RVM 表现更好的那类（背景简单、单人、不需要锁定特定目标），那结论很直接：就用 RVM，别硬上 MatAnyone。把 MatAnyone 留给 RVM 会失败的镜头——背景里有其他人、需要在多目标里只抠一个、长镜头里 RVM 开始乱跳的情况。你手上已经有 SAM 3/3.1，「SAM 出首帧/区域 + RVM 快速软边」对大多数 VR 镜头是性价比最高的组合，MatAnyone 当复杂场景的备选。
