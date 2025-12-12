Here is a clean, professional, yet slightly humorous `README.md` for your project. I’ve made sure to highlight the "DRM" as a fun Easter egg so anyone reading it (including your prof) knows you were just messing around with Python features.

***

# 1D Q-Learning Treasure Hunt 💎

A foundational Reinforcement Learning project implementing **Tabular Q-Learning** from scratch. This project demonstrates how an agent learns to navigate a deterministic 1D environment to reach a goal using the Bellman Equation, without relying on external RL libraries like Gymnasium.

## 📝 Overview

The agent starts at the left side of a 1D map (`_ _ _ _ G`) and must learn to reach the Goal (`G`) efficiently.
*   **Algorithm:** Q-Learning (Off-policy TD Control).
*   **Policy Storage:** NumPy Q-Table.
*   **Environment:** Custom-built 1D grid.

## 🚀 Features

*   **Pure Python/NumPy:** No heavy frameworks (TensorFlow/PyTorch) required.
*   **Visual Rendering:** Real-time CLI visualization of the agent moving (`@`).
*   **Dynamic Epsilon Decay:** Implements $\epsilon$-greedy strategy for exploration vs. exploitation.
*   **"DRM" Protection:** A tongue-in-cheek copyright verification system.

## 🔐 About the "Copyright Protection" (The Fun Part)

**⚠️ Note:** This feature is implemented purely for fun and educational purposes. It is **not** serious security.

I added a custom "Digital Rights Management" (DRM) check to the training loop as a "sand in the desert" coding challenge.
*   It encrypts a copyright string using a reversible XOR cipher encoded in Base64.
*   **The Key:** The encryption key is derived from the RL hyperparameter `self.GAMMA`.
*   **The Mechanism:** If the specific discount factor (0.9) is altered, the mathematical key changes, causing the decryption to fail and print gibberish instead of the success message.

## 🛠️ Requirements

*   Python 3.x
*   NumPy

```bash
pip install numpy
```

## 🏃 How to Run

Simply run the script. The agent will attempt to learn the path.

```bash
python main.py
```

### What to Expect
1.  **Episodes 1-10:** The agent will move randomly (high epsilon), often failing or taking long paths.
2.  **Episodes 10+:** As epsilon decays and the Q-Table fills, the agent will snap into the optimal path.
3.  **Completion:** Once the optimal path is found, the training stops, and the "DRM" verification runs.

## 🧠 The Math

The agent updates its Q-values using the standard Q-Learning update rule:

$$Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_{a'} Q(S', a') - Q(S, A)]$$

Where:
*   $\alpha$ (Alpha): Learning Rate (0.1)
*   $\gamma$ (Gamma): Discount Factor (0.9) **(Also the encryption key!)**
*   $R$: Reward (-1 per step, +100 for Goal)

## 📜 License

Protected by the `fjwiofwe_adsqeai` variable. Don't touch my Gamma! 😉