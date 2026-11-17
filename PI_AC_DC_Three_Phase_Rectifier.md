# PI 控制的三相 AC-DC 整流器 —— 数学与物理过程详解

## 1. 系统概述

本系统是一个**三相电网连接的有源整流器（Active Front End, AFE）**，其核心功能是将三相交流电（AC）转换为可控的直流电（DC）。系统采用 **PI（比例-积分）控制器** 在 dq 同步旋转坐标系下实现对有功和无功功率的解耦控制，从而实现：

- **直流侧电压稳定控制**（DC-link voltage regulation）
- **网侧功率因数校正**（Power Factor Correction, PFC）
- **低谐波电流注入**（Low THD grid current）

### 系统参数（来自图示）

| 参数 | 值 |
|------|------|
| 变压器容量 | 200 kVA |
| 变压器电压比 | 460 V / 400 V |
| 电网侧电压 | 460 V (线电压有效值) |
| 整流器侧电压 | 400 V (线电压有效值) |

---

## 2. 系统物理结构

系统由以下主要部分组成：

```
三相电网 → 变压器 (460V/400V) → 三相PWM整流器 → DC-Link电容 → 直流负载
    ↑                                    ↑
    |                                    |
    +---- 电压/电流传感器 ←---- PI控制器 ←----+
```

### 2.1 AC 电网 (AC Grid)

三相平衡正弦电压源，提供三相对称电压：

$$v_{ga}(t) = V_m \sin(\omega t)$$

$$v_{gb}(t) = V_m \sin(\omega t - \frac{2\pi}{3})$$

$$v_{gc}(t) = V_m \sin(\omega t + \frac{2\pi}{3})$$

其中：
- \( V_m \) 为相电压峰值，\( V_m = \frac{460\sqrt{2}}{\sqrt{3}} \approx 375.6 \text{ V} \)
- \( \omega = 2\pi f = 2\pi \times 60 \approx 377 \text{ rad/s} \)（假设60Hz电网）

### 2.2 变压器 (Transformer)

200 kVA 三相变压器，变比为 460V / 400V。

变压器的作用：
1. **电压匹配**：将电网电压（460V）降低到整流器额定电压（400V）
2. **电气隔离**：提供电网与负载之间的电气隔离
3. **提供滤波电感**：变压器漏抗充当整流器侧的线路电感

变压器模型（简化等值电路，折算到二次侧）：

$$\mathbf{V_2} = \frac{N_2}{N_1} \mathbf{V_1} - (R_{eq} + j\omega L_{eq})\mathbf{I_2}$$

其中：
- \( N_2/N_1 = 400/460 \approx 0.8696 \)
- \( R_{eq} \) 为等效电阻
- \( L_{eq} \) 为等效漏感（关键滤波元件）

### 2.3 三相 PWM 整流器 (Three-Phase Rectifier)

由 **6 个 IGBT 开关**（带反并联二极管）组成的三相全桥拓扑。

每个桥臂的开关函数定义为：

$$S_k = \begin{cases} 1, & \text{上管导通, 下管关断} \\ 0, & \text{上管关断, 下管导通} \end{cases}, \quad k = a, b, c$$

整流器交流侧电压与开关函数的关系：

$$v_{ra} = S_a \cdot V_{dc}$$
$$v_{rb} = S_b \cdot V_{dc}$$
$$v_{rc} = S_c \cdot V_{dc}$$

### 2.4 DC-Link（直流母线）

由直流侧电容 \( C_{dc} \) 和负载电阻 \( R_L \) 组成。

DC-link 电容的动态方程：

$$C_{dc} \frac{dV_{dc}}{dt} = i_{dc,rect} - i_{load}$$

其中：
- \( i_{dc,rect} = S_a \cdot i_{ga} + S_b \cdot i_{gb} + S_c \cdot i_{gc} \) 为整流器输出的直流电流
- \( i_{load} = \frac{V_{dc}}{R_L} \) 为负载电流

---

## 3. 三相系统的数学建模

### 3.1 abc 自然坐标系下的电路方程

对于三相整流器系统，应用基尔霍夫电压定律（KVL）：

$$v_{ga} = R \cdot i_{ga} + L \frac{di_{ga}}{dt} + v_{ra}$$

$$v_{gb} = R \cdot i_{gb} + L \frac{di_{gb}}{dt} + v_{rb}$$

$$v_{gc} = R \cdot i_{gc} + L \frac{di_{gc}}{dt} + v_{rc}$$

其中：
- \( v_{gk} \) 为电网侧相电压 (k = a, b, c)
- \( i_{gk} \) 为电网侧相电流
- \( v_{rk} \) 为整流器交流侧电压
- \( R \) 为线路等效电阻（含变压器铜损）
- \( L \) 为线路等效电感（含变压器漏感）

矩阵形式：

$$\begin{bmatrix} v_{ga} \\ v_{gb} \\ v_{gc} \end{bmatrix} = R \begin{bmatrix} i_{ga} \\ i_{gb} \\ i_{gc} \end{bmatrix} + L \frac{d}{dt} \begin{bmatrix} i_{ga} \\ i_{gb} \\ i_{gc} \end{bmatrix} + \begin{bmatrix} v_{ra} \\ v_{rb} \\ v_{rc} \end{bmatrix}$$

### 3.2 Clark 变换 (abc → αβ)

为了简化三相系统的分析，首先进行 Clark 变换，将三相 abc 坐标变换到两相静止 αβ 坐标系：

$$\begin{bmatrix} f_\alpha \\ f_\beta \end{bmatrix} = \frac{2}{3} \begin{bmatrix} 1 & -\frac{1}{2} & -\frac{1}{2} \\ 0 & \frac{\sqrt{3}}{2} & -\frac{\sqrt{3}}{2} \end{bmatrix} \begin{bmatrix} f_a \\ f_b \\ f_c \end{bmatrix}$$

其中 \( f \) 代表电压或电流。

### 3.3 Park 变换 (αβ → dq)

然后进行 Park 变换，将静止 αβ 坐标系变换到与电网电压同步旋转的 dq 坐标系。

定义 Park 变换矩阵及其逆矩阵：

$$\mathbf{T}_P(\theta) = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}, \qquad \mathbf{T}_P^{-1}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

正变换（αβ → dq）：

$$\begin{bmatrix} f_d \\ f_q \end{bmatrix} = \mathbf{T}_P(\theta) \begin{bmatrix} f_\alpha \\ f_\beta \end{bmatrix}$$

逆变换（dq → αβ）：

$$\begin{bmatrix} f_\alpha \\ f_\beta \end{bmatrix} = \mathbf{T}_P^{-1}(\theta) \begin{bmatrix} f_d \\ f_q \end{bmatrix}$$

其中 \( \theta = \omega t \) 是电网电压矢量的角度，通过 **锁相环 (PLL)** 获取。

**Park 变换的核心优势**：在 dq 坐标系中，稳态下的三相正弦量变为直流量，使得 PI 控制器可以实现零稳态误差跟踪。

### 3.4 从 αβ 方程到 dq 方程的完整推导

这是整个建模过程中**最关键也最容易产生困惑**的一步。下面逐步展示交叉耦合项 \( \omega L i_{gq} \) 和 \( \omega L i_{gd} \) 是如何出现的。

#### 第一步：将 abc 电路方程变换到 αβ 坐标系

abc 坐标系下的矩阵方程为：

$$\mathbf{v}_{g,abc} = R\,\mathbf{i}_{g,abc} + L\,\frac{d\,\mathbf{i}_{g,abc}}{dt} + \mathbf{v}_{r,abc}$$

对等式两边同时左乘 Clark 变换矩阵 \( \mathbf{T}_C \)，由于 \( \mathbf{T}_C \) 是常数矩阵，可以与微分运算交换顺序：

$$\underbrace{\mathbf{T}_C\,\mathbf{v}_{g,abc}}_{\mathbf{v}_{g,\alpha\beta}} = R\,\underbrace{\mathbf{T}_C\,\mathbf{i}_{g,abc}}_{\mathbf{i}_{g,\alpha\beta}} + L\,\frac{d}{dt}\underbrace{(\mathbf{T}_C\,\mathbf{i}_{g,abc})}_{\mathbf{i}_{g,\alpha\beta}} + \underbrace{\mathbf{T}_C\,\mathbf{v}_{r,abc}}_{\mathbf{v}_{r,\alpha\beta}}$$

得到 αβ 坐标系下的电路方程：

$$v_{g\alpha} = R\,i_{g\alpha} + L\,\frac{di_{g\alpha}}{dt} + v_{r\alpha}$$

$$v_{g\beta} = R\,i_{g\beta} + L\,\frac{di_{g\beta}}{dt} + v_{r\beta}$$

> **注意**：Clark 变换矩阵 \( \mathbf{T}_C \) 是常数矩阵（不随时间变化），所以变换后方程的结构与 abc 完全一致——仅仅是把三个方程减少到了两个，**没有产生任何额外的耦合项**。

#### 第二步：将 αβ 方程变换到 dq 坐标系（关键步骤）

将 αβ 方程写成矢量形式：

$$\mathbf{v}_{g,\alpha\beta} = R\,\mathbf{i}_{g,\alpha\beta} + L\,\frac{d\,\mathbf{i}_{g,\alpha\beta}}{dt} + \mathbf{v}_{r,\alpha\beta}$$

利用逆 Park 变换将 dq 量表示为 αβ 量：

$$\mathbf{i}_{g,\alpha\beta} = \mathbf{T}_P^{-1}(\theta)\,\mathbf{i}_{g,dq}$$

代入 αβ 方程：

$$\mathbf{T}_P^{-1}\,\mathbf{v}_{g,dq} = R\,\mathbf{T}_P^{-1}\,\mathbf{i}_{g,dq} + L\,\frac{d}{dt}\!\left[\mathbf{T}_P^{-1}\,\mathbf{i}_{g,dq}\right] + \mathbf{T}_P^{-1}\,\mathbf{v}_{r,dq}$$

#### 第三步：展开微分项（交叉耦合的来源）

**这是最关键的一步。** 由于 \( \mathbf{T}_P^{-1}(\theta) \) 是时间的函数（\( \theta = \omega t \)），对乘积求导必须使用**乘积法则（Product Rule）**：

$$\frac{d}{dt}\!\left[\mathbf{T}_P^{-1}\,\mathbf{i}_{g,dq}\right] = \frac{d\mathbf{T}_P^{-1}}{dt}\,\mathbf{i}_{g,dq} + \mathbf{T}_P^{-1}\,\frac{d\,\mathbf{i}_{g,dq}}{dt}$$

先计算 \( \frac{d\mathbf{T}_P^{-1}}{dt} \)。回顾：

$$\mathbf{T}_P^{-1}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

对每个元素逐一求导，由于 \( \theta = \omega t \)，利用链式法则 \( \frac{d\theta}{dt} = \omega \)：

$$\frac{d}{dt}\cos\theta = -\omega\sin\theta, \quad \frac{d}{dt}\sin\theta = \omega\cos\theta, \quad \frac{d}{dt}(-\sin\theta) = -\omega\cos\theta$$

因此：

$$\frac{d\mathbf{T}_P^{-1}}{dt} = \omega\begin{bmatrix} -\sin\theta & -\cos\theta \\ \cos\theta & -\sin\theta \end{bmatrix}$$

观察这个结果，可以进一步分解为：

$$\frac{d\mathbf{T}_P^{-1}}{dt} = \mathbf{T}_P^{-1} \cdot \omega\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

验证：

$$\mathbf{T}_P^{-1} \cdot \omega\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} 0 & -\omega \\ \omega & 0 \end{bmatrix} = \begin{bmatrix} -\omega\sin\theta & -\omega\cos\theta \\ \omega\cos\theta & -\omega\sin\theta \end{bmatrix} \;\checkmark$$

定义旋转角速度矩阵：

$$\mathbf{J}\omega = \omega\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

于是微分项可以简洁地写为：

$$\frac{d\mathbf{T}_P^{-1}}{dt} = \mathbf{T}_P^{-1} \cdot \mathbf{J}\omega$$

#### 第四步：代入并化简

把展开后的微分项代入原方程：

$$\mathbf{T}_P^{-1}\,\mathbf{v}_{g,dq} = R\,\mathbf{T}_P^{-1}\,\mathbf{i}_{g,dq} + L\!\left[\mathbf{T}_P^{-1}\cdot\mathbf{J}\omega\,\mathbf{i}_{g,dq} + \mathbf{T}_P^{-1}\,\frac{d\,\mathbf{i}_{g,dq}}{dt}\right] + \mathbf{T}_P^{-1}\,\mathbf{v}_{r,dq}$$

等式两边每一项都有公因子 \( \mathbf{T}_P^{-1} \)，**左乘 \( \mathbf{T}_P \) 消去**（因为 \( \mathbf{T}_P\,\mathbf{T}_P^{-1} = \mathbf{I} \)）：

$$\mathbf{v}_{g,dq} = R\,\mathbf{i}_{g,dq} + L\,\frac{d\,\mathbf{i}_{g,dq}}{dt} + L\,\mathbf{J}\omega\,\mathbf{i}_{g,dq} + \mathbf{v}_{r,dq}$$

#### 第五步：展开为标量方程

将矩阵方程展开，注意 \( \mathbf{J}\omega \) 的具体形式：

$$L\,\mathbf{J}\omega\,\mathbf{i}_{g,dq} = L\omega\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} i_{gd} \\ i_{gq} \end{bmatrix} = \begin{bmatrix} -\omega L\,i_{gq} \\ +\omega L\,i_{gd} \end{bmatrix}$$

代入矢量方程的各分量：

**d 轴方程**（第 1 行）：

$$v_{gd} = R\,i_{gd} + L\,\frac{di_{gd}}{dt} - \omega L\,i_{gq} + v_{rd}$$

**q 轴方程**（第 2 行）：

$$v_{gq} = R\,i_{gq} + L\,\frac{di_{gq}}{dt} + \omega L\,i_{gd} + v_{rq}$$

#### 第六步：总结 —— dq 坐标系最终方程

$$\boxed{v_{gd} = R\,i_{gd} + L\,\frac{di_{gd}}{dt} - \omega L\,i_{gq} + v_{rd}}$$

$$\boxed{v_{gq} = R\,i_{gq} + L\,\frac{di_{gq}}{dt} + \omega L\,i_{gd} + v_{rq}}$$

#### 交叉耦合项的物理解释

**为什么会出现 \( -\omega L\,i_{gq} \) 和 \( +\omega L\,i_{gd} \)？**

这些项**不是**来自物理电路本身——在实际的三相电线上并不存在从 d 轴到 q 轴的"物理连接"。它们完全是**数学坐标变换的产物**：

- dq 坐标系以角速度 \( \omega \) 旋转
- 在旋转坐标系中观测一个静止的矢量，会看到它以 \( -\omega \) 反向旋转
- 类似于科里奥利力——在旋转参考系中观察运动物体时出现的"虚拟力"
- 数学上，它来自对 \( \mathbf{T}_P^{-1}(\theta) \) 的时间导数：\( \frac{d\mathbf{T}_P^{-1}}{dt} \) 产生了 \( \omega\mathbf{J} \) 项

> **类比理解**：想象你站在旋转的圆盘上向前直线扔球。从你（旋转坐标系）的视角看，球的轨迹是弯曲的——好像有一个横向的力在"耦合"前后方向和左右方向。这个"力"就是科里奥利效应。dq 方程中的交叉耦合项正是电气领域的"科里奥利效应"。

#### 对比：abc / αβ / dq 三种坐标系下的方程

| 坐标系 | 方程形式 | 交叉耦合 | 稳态量 |
|--------|---------|---------|-------|
| abc (三相静止) | \( v_{gk} = Ri_{gk} + L\frac{di_{gk}}{dt} + v_{rk} \) | 无 | 正弦交变量 |
| αβ (两相静止) | \( v_{g\alpha} = Ri_{g\alpha} + L\frac{di_{g\alpha}}{dt} + v_{r\alpha} \) | 无 | 正弦交变量 |
| dq (两相旋转) | \( v_{gd} = Ri_{gd} + L\frac{di_{gd}}{dt} - \omega Li_{gq} + v_{rd} \) | **有**（ωL 项） | **直流常量** |

**关键取舍**：dq 变换用"引入交叉耦合项"的代价，换来了"稳态量变为直流"的巨大优势。而交叉耦合项可以通过控制器中的**前馈解耦**来消除（见第 5 节）。

### 3.5 dq 坐标系中电网电压的对齐

采用**电网电压定向 (Grid Voltage Oriented, GVO)** 策略：

将 d 轴对齐到电网电压矢量方向，则：

$$v_{gd} = V_g, \quad v_{gq} = 0$$

其中 \( V_g \) 为电网电压幅值。

在此定向下：
- **有功功率**仅与 d 轴电流相关
- **无功功率**仅与 q 轴电流相关

---

## 4. 功率分析

### 4.1 瞬时功率理论

在 dq 坐标系下，三相系统的有功功率和无功功率为：

$$\boxed{P = \frac{3}{2}(v_{gd} \cdot i_{gd} + v_{gq} \cdot i_{gq})}$$

$$\boxed{Q = \frac{3}{2}(v_{gq} \cdot i_{gd} - v_{gd} \cdot i_{gq})}$$

由于 \( v_{gq} = 0 \)（电压定向），简化为：

$$P = \frac{3}{2} V_g \cdot i_{gd}$$

$$Q = -\frac{3}{2} V_g \cdot i_{gq}$$

**物理意义**：
- \( i_{gd} \)（d 轴电流）**直接控制有功功率传输**
- \( i_{gq} \)（q 轴电流）**直接控制无功功率传输**

#### 4.1.1 为什么 \( i_{gq}^* = 0 \) 能实现单位功率因数？

功率因数定义为 \( \text{PF} = \cos\varphi \)，其中 \( \varphi \) 是电压与电流之间的相位差。单位功率因数（PF=1）等价于 \( Q = 0 \)。

在电压定向下（\( v_{gq} = 0 \)）：\( Q = -\frac{3}{2} V_g \cdot i_{gq} \)。由于 \( V_g \neq 0 \)，当且仅当 \( i_{gq} = 0 \) 时 \( Q = 0 \)。

从相量角度看：d 轴对齐电压方向，电流矢量的 q 分量就是相对于电压的"偏转量"。\( i_{gq} = 0 \) 意味着电流完全对准电压方向，两者同相，功率因数为 1。

### 4.2 功率平衡方程

#### 4.2.1 DC-link 能量守恒——动态方程的推导

DC-link 电容中存储的能量为 \( E_C = \frac{1}{2} C_{dc} V_{dc}^2 \)。对时间求导（能量变化率 = 净输入功率）：

$$\frac{dE_C}{dt} = \frac{1}{2} C_{dc} \frac{dV_{dc}^2}{dt} = P_{\text{in}} - P_{\text{out}}$$

其中 \( P_{\text{in}} = \frac{3}{2} V_g \cdot i_{gd} \)（AC 侧输入功率），\( P_{\text{out}} = \frac{V_{dc}^2}{R_L} \)（DC 负载功率），得到：

$$\frac{1}{2} C_{dc} \frac{dV_{dc}^2}{dt} = \frac{3}{2} V_g \cdot i_{gd} - \frac{V_{dc}^2}{R_L}$$

#### 4.2.2 稳态条件与动态方程的关系

> **\( P_{ac} = P_{dc} \) 仅在稳态下成立，不是任何时刻都成立。** 该方程描述的是动态过程：稳态时 \( P_{\text{in}} = P_{\text{out}} \)，\( dV_{dc}/dt = 0 \)；负载突增时 \( P_{\text{in}} < P_{\text{out}} \)，\( V_{dc} \) 下降；负载突减时 \( P_{\text{in}} > P_{\text{out}} \)，\( V_{dc} \) 上升。控制器的任务就是调整 \( i_{gd} \) 使系统回到稳态。

#### 4.2.3 稳态功率平衡

当 \( dV_{dc}/dt = 0 \) 时：

$$\frac{3}{2} V_g \cdot i_{gd,\text{ss}} = \frac{V_{dc,\text{ss}}^2}{R_L}$$

---

## 5. PI 控制器设计

系统采用 **级联控制结构（Cascaded Control）**，包含：
1. **外环**：DC-link 电压控制环（慢环）
2. **内环**：电网电流控制环（快环）

### 5.1 控制系统结构图

```
VdcRef ──→ [PI_v] ──→ id_ref ──→ [PI_id] ──→ vrd* ──→ [PWM] ──→ 整流器
  ↑                                  ↑
  |                                  |
VdcF ←── DC电压反馈          igd ←── d轴电流反馈

                     0 ──→ [PI_iq] ──→ vrq* ──→ [PWM] ──→ 整流器
                              ↑
                              |
                       igq ←── q轴电流反馈
```

### 5.2 内环：dq 电流控制器

#### 5.2.0 控制系统的输入、输出与可控变量分析

| 变量角色 | 变量 | 说明 |
|---------|------|------|
| **控制输入** | \( v_{rd}, v_{rq} \)（整流器电压） | 通过 PWM 改变 IGBT 开关占空比产生 |
| **被控输出** | 内环：\( i_{gd}, i_{gq} \)；外环：\( V_{dc} \) | 控制目标 |
| **可测扰动** | \( v_{gd}, v_{gq} \)（电网电压） | 通过前馈补偿 |
| **耦合扰动** | \( \omega L i_{gq}, \omega L i_{gd} \) | 通过解耦前馈消除 |

\( v_{rd} \) 可作为控制输入，因为 PWM 整流器是**功率放大器**：控制器计算参考电压 \( v_{rd}^* \)，经反 Park 变换回三相，PWM 调制器将其转化为 IGBT 开关信号，使桥臂平均输出电压等于参考值。

#### 5.2.1 控制目标

将 dq 方程整理为"控制输入→被控输出"的因果形式：

$$L \frac{di_{gd}}{dt} = \underbrace{v_{gd}}_{\text{可测扰动}} - R \cdot i_{gd} + \underbrace{\omega L \cdot i_{gq}}_{\text{耦合扰动}} - \underbrace{v_{rd}}_{\text{控制输入}}$$

$$L \frac{di_{gq}}{dt} = \underbrace{v_{gq}}_{\text{可测扰动}} - R \cdot i_{gq} - \underbrace{\omega L \cdot i_{gd}}_{\text{耦合扰动}} - \underbrace{v_{rq}}_{\text{控制输入}}$$

#### 5.2.2 PI 控制器 + 前馈解耦

设计思想：把能测量到的扰动用前馈补偿掉，剩下的纯误差用 PI 处理。

d 轴：

$$\boxed{v_{rd}^* = \underbrace{v_{gd}}_{\text{电网前馈}} + \underbrace{\omega L \cdot i_{gq}}_{\text{解耦前馈}} - \underbrace{\left(K_{pi} + \frac{K_{ii}}{s}\right)(i_{gd}^* - i_{gd})}_{\text{PI 控制}}}$$

q 轴：

$$\boxed{v_{rq}^* = \underbrace{v_{gq}}_{\text{电网前馈}} - \underbrace{\omega L \cdot i_{gd}}_{\text{解耦前馈}} - \underbrace{\left(K_{pi} + \frac{K_{ii}}{s}\right)(i_{gq}^* - i_{gq})}_{\text{PI 控制}}}$$

将控制律代入物理方程后，\( v_{gd} \) 和 \( \omega L i_{gq} \) 正好对消，d 轴变为独立的一阶 RL 系统被 PI 驱动：\( L \frac{di_{gd}}{dt} + R i_{gd} = \text{PI}(i_{gd}^* - i_{gd}) \)。

参数说明：
- \( K_{pi} \)：电流环比例增益
- \( K_{ii} \)：电流环积分增益
- \( v_{gd}, v_{gq} \)：电网电压前馈项（抵消电网电压扰动）
- \( \omega L \cdot i_{gq}, \omega L \cdot i_{gd} \)：交叉耦合解耦项

#### 5.2.3 PI 控制器传递函数

电流环 PI 控制器的传递函数：

$$G_{PI,i}(s) = K_{pi} + \frac{K_{ii}}{s} = K_{pi} \cdot \frac{s + K_{ii}/K_{pi}}{s} = K_{pi} \cdot \frac{s + \tau_i^{-1}}{s}$$

其中 \( \tau_i = K_{pi}/K_{ii} \) 为积分时间常数。

#### 5.2.4 解耦后的电流环传递函数

在完美解耦和前馈补偿下，d 轴和 q 轴的电流控制环变为相同的一阶系统：

被控对象（Plant）：

$$G_p(s) = \frac{I_{gd}(s)}{V_{rd}(s)} = \frac{1}{Ls + R}$$

开环传递函数：

$$G_{OL,i}(s) = G_{PI,i}(s) \cdot G_p(s) = \frac{K_{pi}(s + K_{ii}/K_{pi})}{s} \cdot \frac{1}{Ls + R}$$

#### 5.2.5 电流环参数整定

采用**零极点对消法**（Pole-Zero Cancellation）：

令 PI 控制器的零点对消被控对象的极点：

$$\frac{K_{ii}}{K_{pi}} = \frac{R}{L}$$

则开环传递函数简化为：

$$G_{OL,i}(s) = \frac{K_{pi}}{Ls}$$

闭环传递函数为一阶系统：

$$G_{CL,i}(s) = \frac{K_{pi}/L}{s + K_{pi}/L} = \frac{1}{\tau_{ci} s + 1}$$

其中闭环时间常数：

$$\boxed{\tau_{ci} = \frac{L}{K_{pi}}}$$

选择期望带宽 \( \omega_{bw,i} \)，则：

$$K_{pi} = L \cdot \omega_{bw,i}$$

$$K_{ii} = R \cdot \omega_{bw,i}$$

典型设计：电流环带宽选取为开关频率的 1/10 至 1/5。

### 5.3 外环：DC-Link 电压控制器

#### 5.3.1 DC-Link 动态模型与小信号线性化

DC-link 能量平衡（非线性方程）：

$$C_{dc} V_{dc} \frac{dV_{dc}}{dt} = \frac{3}{2} V_g \cdot i_{gd} - P_{\text{load}}$$

**小信号线性化步骤**：

**第一步**：定义稳态工作点 + 小扰动：\( V_{dc} = V_{dc0} + \tilde{v}_{dc} \)，\( i_{gd} = I_{gd0} + \tilde{i}_{gd} \)

**第二步**：代入并展开左边的乘积（注意 \( \frac{dV_{dc}}{dt} = \frac{d\tilde{v}_{dc}}{dt} \)）：

$$C_{dc}(V_{dc0} + \tilde{v}_{dc})\frac{d\tilde{v}_{dc}}{dt} = C_{dc} V_{dc0} \frac{d\tilde{v}_{dc}}{dt} + \underbrace{C_{dc} \tilde{v}_{dc} \frac{d\tilde{v}_{dc}}{dt}}_{\text{二阶小量，忽略}}$$

> **为什么 \( V_{dc} \) 变成了 \( V_{dc0} \) 而不是 \( V_{dc0} + \tilde{v}_{dc} \)？** 不是粗暴替换！而是展开后，\( \tilde{v}_{dc} \cdot \frac{d\tilde{v}_{dc}}{dt} \) 是两个小量相乘（二阶小量），远小于 \( V_{dc0} \cdot \frac{d\tilde{v}_{dc}}{dt} \)（一阶小量），因此被忽略。

**第三步**：分离稳态方程（左边=0）和小信号方程：

$$\boxed{C_{dc} V_{dc0} \frac{d\tilde{v}_{dc}}{dt} = \frac{3}{2} V_g \cdot \tilde{i}_{gd} - \tilde{P}_{\text{load}}}$$

**第四步**：求传递函数（令 \( \tilde{P}_{\text{load}} = 0 \)）：

$$G_{v}(s) = \frac{\tilde{V}_{dc}(s)}{\tilde{I}_{gd}(s)} = \frac{3 V_g}{2 C_{dc} V_{dc0} \cdot s}$$

这是一个纯积分环节：增加 \( i_{gd} \) → 多注入功率 → 充电电容 → \( V_{dc} \) 持续上升。

#### 5.3.2 电压环 PI 控制器

电压环 PI 控制器：

$$G_{PI,v}(s) = K_{pv} + \frac{K_{iv}}{s}$$

其输出为 d 轴电流参考值 \( i_{gd}^* \)：

$$\boxed{i_{gd}^* = \left(K_{pv} + \frac{K_{iv}}{s}\right)(V_{dc}^* - V_{dc})}$$

#### 5.3.3 电压环开环传递函数

考虑内环闭环传递函数 \( G_{CL,i}(s) \)：

$$G_{OL,v}(s) = G_{PI,v}(s) \cdot G_{CL,i}(s) \cdot G_v(s)$$

$$= \left(K_{pv} + \frac{K_{iv}}{s}\right) \cdot \frac{1}{\tau_{ci} s + 1} \cdot \frac{3 V_g}{2 C_{dc} V_{dc0} \cdot s}$$

#### 5.3.4 电压环参数整定

若内环足够快（\( \tau_{ci} \ll \tau_{cv} \)），可近似 \( G_{CL,i}(s) \approx 1 \)：

$$G_{OL,v}(s) \approx \frac{(K_{pv} s + K_{iv}) \cdot 3 V_g}{2 C_{dc} V_{dc0} \cdot s^2}$$

采用**对称最优法**（Symmetric Optimum）或**模量最优法**（Modulus Optimum）进行参数整定：

**模量最优法整定结果**：

$$K_{pv} = \frac{2 C_{dc} V_{dc0} \omega_{bw,v}}{3 V_g}$$

$$K_{iv} = \frac{K_{pv}}{\tau_{iv}}$$

其中 \( \omega_{bw,v} \) 为电压环期望带宽，通常选取为电流环带宽的 1/5 至 1/10，以保证级联控制的稳定性。

### 5.4 带宽分离原则

$$\boxed{\omega_{bw,v} \ll \omega_{bw,i} \ll \omega_{sw}}$$

- \( \omega_{sw} \)：PWM 开关角频率
- \( \omega_{bw,i} \)：电流环带宽（典型值：开关频率的 1/10 ~ 1/5）
- \( \omega_{bw,v} \)：电压环带宽（典型值：电流环带宽的 1/5 ~ 1/10）

---

## 6. 锁相环 (PLL)

锁相环用于从电网电压中提取相角 \( \theta \)，这是 Park 变换的基础。

### 6.1 SRF-PLL 结构

$$v_{gq} = V_g \sin(\theta_{grid} - \theta_{PLL}) \approx V_g (\theta_{grid} - \theta_{PLL})$$

当 \( \theta_{PLL} \) 锁定到 \( \theta_{grid} \) 时，\( v_{gq} \to 0 \)。

PLL 的控制律：

$$\hat{\omega} = K_{p,PLL} \cdot v_{gq} + K_{i,PLL} \int v_{gq} \, dt$$

$$\hat{\theta} = \int \hat{\omega} \, dt$$

---

## 7. PWM 调制

### 7.1 空间矢量脉宽调制 (SVPWM)

将 dq 坐标系的参考电压 \( v_{rd}^*, v_{rq}^* \) 经过反 Park 变换和反 Clark 变换，得到三相调制信号，再通过 SVPWM 生成 IGBT 的门极驱动信号。

反 Park 变换：

$$\begin{bmatrix} v_{r\alpha}^* \\ v_{r\beta}^* \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} v_{rd}^* \\ v_{rq}^* \end{bmatrix}$$

调制指数：

$$m = \frac{\sqrt{v_{rd}^{*2} + v_{rq}^{*2}}}{V_{dc}/2}$$

线性调制范围：\( 0 \leq m \leq 1.15 \)（SVPWM 情况下）

---

## 8. DC-Link 电压滤波

图中可以看到 DC-link 电压反馈信号 `[VdcF]`，这表示使用了低通滤波器对测量的 DC 电压进行滤波。

滤波器传递函数（一阶低通）：

$$G_{filter}(s) = \frac{1}{\tau_f s + 1}$$

其中 \( \tau_f \) 为滤波时间常数。这可以抑制 DC-link 电压中的开关纹波和二次谐波分量。

---

## 9. 完整控制信号流程

整理整个系统从信号采集到开关驱动的完整信号流程：

1. **测量**：采集三相电网电压 \( v_{gABC} \)、三相电网电流 \( i_{gABC} \)、DC-link 电压 \( V_{dc} \)

2. **PLL 锁相**：从 \( v_{gABC} \) 提取电网相角 \( \theta \)

3. **坐标变换**：
   - \( v_{gABC} \xrightarrow{abc \to dq} v_{gd}, v_{gq} \)
   - \( i_{gABC} \xrightarrow{abc \to dq} i_{gd}, i_{gq} \)

4. **DC 电压滤波**：\( V_{dc} \xrightarrow{LPF} V_{dcF} \)

5. **外环 PI**：\( e_v = V_{dc}^* - V_{dcF} \xrightarrow{PI_v} i_{gd}^* \)

6. **设定 q 轴电流参考**：\( i_{gq}^* = 0 \)（单位功率因数）

7. **内环 PI + 解耦**：
   - \( v_{rd}^* = v_{gd} + \omega L \cdot i_{gq} - PI_d(i_{gd}^* - i_{gd}) \)
   - \( v_{rq}^* = v_{gq} - \omega L \cdot i_{gd} - PI_q(i_{gq}^* - i_{gq}) \)

8. **反坐标变换**：\( v_{rd}^*, v_{rq}^* \xrightarrow{dq \to abc} v_{ra}^*, v_{rb}^*, v_{rc}^* \)

9. **PWM 调制**：生成 6 路 IGBT 门极信号 → GateR（Gate 驱动模块）

10. **功率转换**：三相整流器桥将 AC 功率转换为 DC 功率

---

## 10. 关键方程汇总

### 电路方程 (dq 坐标系)

$$v_{gd} = R \cdot i_{gd} + L \frac{di_{gd}}{dt} - \omega L \cdot i_{gq} + v_{rd}$$

$$v_{gq} = R \cdot i_{gq} + L \frac{di_{gq}}{dt} + \omega L \cdot i_{gd} + v_{rq}$$

### DC-Link 动态

$$C_{dc} \frac{dV_{dc}}{dt} = \frac{3}{2} \frac{V_g \cdot i_{gd}}{V_{dc}} - \frac{V_{dc}}{R_L}$$

### 功率方程

$$P = \frac{3}{2} V_g \cdot i_{gd}, \quad Q = -\frac{3}{2} V_g \cdot i_{gq}$$

### 电流环控制律

$$v_{rd}^* = v_{gd} + \omega L \cdot i_{gq} - K_{pi}(i_{gd}^* - i_{gd}) - K_{ii}\int(i_{gd}^* - i_{gd})dt$$

$$v_{rq}^* = v_{gq} - \omega L \cdot i_{gd} - K_{pi}(i_{gq}^* - i_{gq}) - K_{ii}\int(i_{gq}^* - i_{gq})dt$$

### 电压环控制律

$$i_{gd}^* = K_{pv}(V_{dc}^* - V_{dc}) + K_{iv}\int(V_{dc}^* - V_{dc})dt$$

---

## 11. 系统稳定性分析

### 11.1 电流环闭环特征方程

零极点对消后：

$$1 + \frac{K_{pi}}{Ls} = 0 \implies s = -\frac{K_{pi}}{L}$$

极点始终在左半平面 → **电流环无条件稳定**

### 11.2 电压环稳定性

电压环的相位裕度（Phase Margin, PM）和增益裕度（Gain Margin, GM）需满足：

$$PM > 45°, \quad GM > 6 \text{ dB}$$

带宽分离是保证级联系统稳定性的关键条件。

---

## 12. 图中各模块对应关系

| 图中模块 | 功能 | 对应数学模型 |
|----------|------|-------------|
| AC grid | 三相交流电源 | \( v_{ga}, v_{gb}, v_{gc} \) |
| 200 kVA 变压器 | 电压变换与隔离 | 匝比 460/400, 漏感 \( L \) |
| Three-phase rectifier | 功率变换 | 开关函数 \( S_a, S_b, S_c \) |
| DC Link | 能量缓冲 | \( C_{dc} dV_{dc}/dt \) |
| Rectifier control | PI 控制器 + 解耦 | 电压环 + 电流环 PI |
| GateR | 门极驱动 | PWM 信号生成 |
| VdcRef | DC 电压参考值 | \( V_{dc}^* \) |
| VdcF | 滤波后的 DC 电压 | \( G_{filter}(s) \cdot V_{dc} \) |
| VgABC | 三相电网电压测量 | \( v_{ga}, v_{gb}, v_{gc} \) |
| IgABC | 三相电网电流测量 | \( i_{ga}, i_{gb}, i_{gc} \) |
| Scopes | 示波器/可视化 | 监控波形 |
| NoOp | 无操作/信号传递 | 信号透传 |
| f(x) = 0 | 初始条件/信号源 | 系统初始化 |
| SR (Rectifier control输出) | 开关控制信号 | PWM 调制信号 |

---

## 13. 总结

本系统通过以下核心思想实现高性能 AC-DC 功率转换：

1. **dq 坐标系变换**：将时变的三相交流量转化为稳态直流量，使经典 PI 控制器可以实现无差跟踪。

2. **电压定向控制 (VOC)**：将 d 轴对齐到电网电压矢量方向，实现有功/无功功率的解耦控制。

3. **级联 PI 控制**：
   - 外环 PI 控制直流电压，输出 d 轴电流参考
   - 内环 PI 控制 dq 轴电流，输出整流器电压参考

4. **前馈解耦**：消除 dq 轴之间的交叉耦合，实现 d/q 轴独立控制。

5. **带宽分离**：内环（电流环）远快于外环（电压环），保证级联系统的稳定性和良好的动态响应。
