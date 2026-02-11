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

然后进行 Park 变换，将静止 αβ 坐标系变换到与电网电压同步旋转的 dq 坐标系：

$$\begin{bmatrix} f_d \\ f_q \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} f_\alpha \\ f_\beta \end{bmatrix}$$

其中 \( \theta = \omega t \) 是电网电压矢量的角度，通过 **锁相环 (PLL)** 获取。

**Park 变换的核心优势**：在 dq 坐标系中，稳态下的三相正弦量变为直流量，使得 PI 控制器可以实现零稳态误差跟踪。

### 3.4 dq 坐标系下的系统方程

在 dq 同步旋转坐标系中，三相电路方程变为：

$$\boxed{v_{gd} = R \cdot i_{gd} + L \frac{di_{gd}}{dt} - \omega L \cdot i_{gq} + v_{rd}}$$

$$\boxed{v_{gq} = R \cdot i_{gq} + L \frac{di_{gq}}{dt} + \omega L \cdot i_{gd} + v_{rq}}$$

**关键特征**：
- 出现了 **交叉耦合项**：\( -\omega L \cdot i_{gq} \) 和 \( +\omega L \cdot i_{gd} \)
- d 轴和 q 轴的电流动态相互耦合
- 需要**前馈解耦**来实现独立控制

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
- 设 \( i_{gq}^* = 0 \) 即可实现**单位功率因数运行**

### 4.2 功率平衡方程

在稳态下，忽略损耗时：

$$P_{ac} = P_{dc}$$

$$\frac{3}{2} V_g \cdot i_{gd} = V_{dc} \cdot I_{dc} = \frac{V_{dc}^2}{R_L}$$

DC-link 的能量守恒：

$$\frac{1}{2} C_{dc} \frac{dV_{dc}^2}{dt} = P_{ac} - P_{dc} = \frac{3}{2} V_g \cdot i_{gd} - \frac{V_{dc}^2}{R_L}$$

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

#### 5.2.1 控制目标

将 dq 坐标系下的系统方程重新整理为控制形式：

$$L \frac{di_{gd}}{dt} = v_{gd} - R \cdot i_{gd} + \omega L \cdot i_{gq} - v_{rd}$$

$$L \frac{di_{gq}}{dt} = v_{gq} - R \cdot i_{gq} - \omega L \cdot i_{gd} - v_{rq}$$

#### 5.2.2 PI 控制器 + 前馈解耦

d 轴整流器参考电压：

$$\boxed{v_{rd}^* = v_{gd} + \omega L \cdot i_{gq} - \left(K_{pi} + \frac{K_{ii}}{s}\right)(i_{gd}^* - i_{gd})}$$

q 轴整流器参考电压：

$$\boxed{v_{rq}^* = v_{gq} - \omega L \cdot i_{gd} - \left(K_{pi} + \frac{K_{ii}}{s}\right)(i_{gq}^* - i_{gq})}$$

其中：
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

#### 5.3.1 DC-Link 动态模型

DC-link 电容上的能量平衡：

$$C_{dc} V_{dc} \frac{dV_{dc}}{dt} = P_{in} - P_{out} = \frac{3}{2} V_g \cdot i_{gd} - P_{load}$$

对 \( V_{dc} \) 进行小信号线性化（在工作点 \( V_{dc0} \) 附近）：

$$V_{dc} = V_{dc0} + \tilde{v}_{dc}$$

$$i_{gd} = I_{gd0} + \tilde{i}_{gd}$$

线性化后的小信号模型：

$$C_{dc} V_{dc0} \frac{d\tilde{v}_{dc}}{dt} = \frac{3}{2} V_g \cdot \tilde{i}_{gd} - \tilde{P}_{load}$$

DC-link 电压到 d 轴电流的传递函数（\( \tilde{P}_{load} = 0 \)）：

$$G_{v}(s) = \frac{\tilde{V}_{dc}(s)}{\tilde{I}_{gd}(s)} = \frac{3 V_g}{2 C_{dc} V_{dc0} \cdot s}$$

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
