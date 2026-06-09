Prompt: Implement D1-BN Embedding + Sigmoid Rejection Router
Objective
Implement a domain routing system for 3 domains (D1, D2, D3) using a frozen D1-trained batch normalization encoder as a shared representation space, and a lightweight classifier head to distinguish D2 vs D3, with rejection for D1.
The key idea is:
Use D1 BatchNorm as a fixed normalization reference space, and learn how D2 and D3 deviate from this space.
1. Model Components
1.1 Shared Backbone (Frozen)
You are given a pretrained model trained on D1.
It consists of:
Shared CNN feature extractor
BatchNorm layers trained on D1: BN_D1
Requirement:
Freeze ALL parameters in:
CNN backbone
BN_D1 (very important: running mean/variance must NOT update)
So during training:
for param in backbone.parameters():
    param.requires_grad = False
backbone.eval()  # ensures BN_D1 uses fixed statistics
1.2 Embedding Function
For any input audio sample x from domain D2 or D3:
x → Shared CNN → BN_D1 → Global Pool → 2048-d embedding z
Define:
z=fD1(x)z = f_{D1}(x)z=fD1(x)
Where:
BN_D1 acts as a fixed “reference normalization space”
Output is a 2048-dimensional embedding vector
Implementation detail:
Use global average pooling after CNN+BN
Flatten into (batch_size, 2048)
1.3 Router Head (Trainable)
A lightweight classifier that maps embeddings to domain probabilities.
Architecture:
Linear(2048 → 128)
ReLU
Linear(128 → 2)
Sigmoid (or logits for BCEWithLogitsLoss)
Output:
p = [p_D2, p_D3]
Define:
self.router = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)
2. Training Setup
2.1 Training Data
Only use:
D2 labeled as class [1, 0]
D3 labeled as class [0, 1]
Do NOT use D1 samples for training the router.
2.2 Forward Pass
For each batch:
with torch.no_grad():
    z = backbone(x)   # includes BN_D1
Then:
logits = router(z)
2.3 Loss Function
Use:
BCEWithLogitsLoss
Target format:
D2 → [1, 0]
D3 → [0, 1]
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, labels)
2.4 Optimization
Only update router parameters:
optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)
3. Intuition (Important for implementation correctness)
BN_D1 acts as a fixed normalization coordinate system:
It assumes statistics of D1 (mean ≈ 0, variance ≈ 1 in feature space)
When D2/D3 pass through it:
Their feature distributions shift differently in embedding space
So:
BN_D1 transforms all domains into a D1-aligned feature space
The router learns:
direction of shift = domain identity
4. Inference Logic
At test time, for any input x:
z = backbone(x)
logits = router(z)
probs = sigmoid(logits)
p2, p3 = probs[0], probs[1]
Then apply rejection mechanism:
if max(p2, p3) < threshold:
    domain = "D1"   # rejection class
elif p2 > p3:
    domain = "D2"
else:
    domain = "D3"
Recommended:
threshold = 0.5 (tune on validation set)
5. Key Implementation Constraints
BN_D1 MUST remain frozen (no running stats updates)
Backbone MUST remain frozen
Only router is trainable
No D1 samples are used in router training
Output layer uses logits (recommended for stability)
6. Expected Behavior
D2 vs D3 separation is learned in BN_D1-aligned space
D1 acts as implicit “out-of-distribution” class via low confidence
Router learns domain shift rather than absolute features