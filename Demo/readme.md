# Overfit Swarm Engine 🌪️

**100,000 AI Agents. 11ms Inference. Zero Main-Thread Bottlenecks.**

![Overfit Swarm Demo](https://img.shields.io/badge/Scale-100k_Units-blue)
![Inference Time](https://img.shields.io/badge/Inference-11ms-brightgreen)
![Tech Stack](https://img.shields.io/badge/Tech-.NET_10_%7C_Unity_6-lightgrey)

## 📖 What is it?
The **Overfit Swarm Engine** is an experimental, high-performance distributed AI simulation framework. It demonstrates how to bypass traditional game engine limitations by offloading massive-scale calculations to a dedicated, highly optimized external server.

Instead of choking the Unity main thread with physics and AI logic for 100,000 agents, the Overfit Engine runs a standalone `.NET` process that streams calculated steering forces back to Unity in real-time.

## 🤔 Why build this? (The Motivation)
Standard approaches to swarm intelligence in game engines (like `MonoBehaviour.Update` or even basic Job Systems) often hit a wall when scaling past 10,000 units. 

The goal of this project was to answer one question: **Can we achieve fluid, emergent AI behavior at a massive scale (100k+) while keeping the client application running at maximum framerate?**

The answer is yes, achieved through:
1. **Evolutionary AI:** The swarm is not hardcoded (no simple `MoveTowards`). It uses a neural network trained from scratch via a Genetic Algorithm to track targets and avoid predators, naturally developing a "swirl/tornado" behavior.
2. **Zero-Allocation Networking:** A custom binary TCP protocol utilizing `Span<T>`, `MemoryMarshal`, and `unsafe` C# pointers to transfer megabytes of state data with zero Garbage Collection (GC) overhead.
3. **GPU Instancing:** Rendering 100,000 units in a single draw call using Unity's `Graphics.RenderMeshInstanced`.

## 🚀 Key Features
* **Dual-Mode Execution:**
  * **Training Mode:** Runs an evolutionary Genetic Algorithm, applying selection pressure, mutations, and saving the best genomic weights to a `.bin` file.
  * **Demo Mode (Inference):** Loads the pre-trained `.bin` weights for ultra-fast, static inference.
* **Microsecond Math:** Custom matrix operations utilizing native math and C# `MemoryMarshal.Cast` for blazing-fast array manipulation.
* **Emergent Behavior:** Bots organically form cohesive structures, orbit targets, and scatter when a predator approaches.

## 🏗️ Architecture Overview

The system is split into two distinct parts:

1. **The Brain (C# .NET Server)**
   * Acts as the ultimate authority on physics and AI.
   * Receives target/predator positions from the client.
   * Processes the neural network for all 100k bots.
   * Streams back raw X/Z positions.
   
2. **The Visualizer (Unity Client)**
   * Extremely lightweight.
   * Only reads user input (moving the target/predator).
   * Receives the byte stream, writes directly to GPU matrices using `unsafe` code, and renders the frame.

## 🎮 How to Run

### 1. Start the Server
Navigate to the server project folder and run the following commands in your terminal:

To run the pre-trained, high-performance demo:
`dotnet run -- demo`

To start training a new swarm from scratch:
`dotnet run -- training`

### 2. Start the Client
1. Open the Unity project.
2. Ensure **Allow 'unsafe' Code** is checked in `Player Settings`.
3. Open the `SampleScene` and hit **Play**.
4. Move the `Target` object to watch the swarm react instantly.

## 🧠 The AI Model (Under the Hood)
Each bot acts based on a 10-parameter genomic array (weights and biases). The network normalizes distance to the target and predator threat levels, passing them through a `MathF.Tanh` activation function to output bounded `[-1, 1]` steering accelerations.

During training, bots are rewarded for tangential motion (orbiting) and penalized heavily (fitness death) for flying out of bounds or touching a predator.

## 📜 License
Copyright (c) 2026 DevOnBike.
This project is part of DevonBike Overfit and is licensed under the GNU AGPLv3. For commercial licensing options, contact: devonbike@gmail.com