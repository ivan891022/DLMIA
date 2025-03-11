# **Deep Learning in Medical Image Analysis**

本系列作業聚焦於不同醫學影像的深度學習應用，涵蓋圖像分類、定位、分割等多種模型架構與訓練策略。

---



## **Lab 1: 手寫數字辨識**

### **目標**
- 訓練深度學習模型來分類 **手寫數字 (0-9)**。

### **技術與方法**
- **Python**: `pandas`, `numpy`, `torch`
- **深度學習框架**: `PyTorch`
- **數據處理**:
  - 使用 `sklearn.model_selection.train_test_split` 將數據分為訓練集與測試集。
  - 透過 `matplotlib` 和 `plotly` 進行數據可視化。
- **模型架構**:
  - `SimpleNN`：全連接神經網路 (Fully Connected Layers)。
  - 採用 `ReLU`、`BatchNorm`、`Dropout` 等技術。
- **訓練與評估**:
  - **損失函數**: `CrossEntropyLoss`
  - **優化器**: `Adam`
  - **評估指標**: `accuracy_score`, `f1_score`
- **測試與結果**:
  - 對測試數據進行推論。
  - 結果輸出為 CSV (`Hw1_submission_Z.csv`)。

---

## **Lab 2: 醫學影像分類**

### **目標**
- 訓練與比較不同深度學習模型在 **醫學影像分類** 任務上的表現。

### **技術與方法**
- **深度學習框架**: `PyTorch`
- **模型架構**:
  - **VGG16**:
    - 使用 **預訓練 VGG16** 進行特徵提取。
    - 修改 `MaxPool2d` 為 `Identity()`，以適應醫學影像。
  - **ResNet50**:
    - 使用 **預訓練 ResNet50**，並調整 `AdaptiveAvgPool2d`。
    - 提取 **2048維特徵**，與 **年齡和性別** 拼接後進行分類。
  - **Swin Transformer (Swin ViT)**:
    - 設定影像尺寸 `56x56`，並移除 `head.fc` 進行特徵提取。
- **數據處理**:
  - **三通道轉換**: 使用 **前一張、當前張、後一張** 拼接為輸入。
  - **數據增強**: 確保醫學影像的診斷信息不受影響。
- **訓練與評估**:
  - **損失函數**: `CrossEntropyLoss`
  - **優化器**: `Adam`
  - **學習率調整**與 **Early Stopping** 避免過擬合。

---

## **Lab 3: 影像融合技術於醫學影像分類**

### **目標**
- 比較 **ResNet50** 與 **3D CNN**、**Early Fusion**、**Late Fusion**、**Single Slice** 方法在 **醫學影像分類** 上的效能。

### **技術與方法**
- **模型架構**:
  - **ResNet50**: 用於 **2D 影像分類**。
  - **3D CNN**: 處理 **三維醫學影像**。
  - **Early Fusion**: **輸入端融合**，將多個影像切片組合為單一輸入。
  - **Late Fusion**: **結果端融合**，多個 ResNet50 模型輸出加權。
  - **Single Slice**: 只使用 **單張影像** 進行分類。
- **訓練與測試**:
  - **損失函數**: `CrossEntropyLoss`
  - **優化器**: `Adam`
  - **評估指標**: `Accuracy`, `F1 Score`, `ROC-AUC Score`

---

## **Lab 4: 醫學影像分割**

### **目標**
- 探討 **U-Net** 與 **FCN-8s** 在醫學影像分割任務中的應用。

### **技術與方法**
- **模型架構**:
  - **U-Net**:
    - Encoder：多層卷積與池化，提取多尺度特徵。
    - Decoder：上採樣恢復空間解析度。
  - **FCN-8s**:
    - Encoder：基於 VGG-16 進行特徵提取。
    - Decoder：多層上採樣，結合不同尺度的特徵。
- **訓練與評估**:
  - **損失函數**: `Dice Loss`, `CrossEntropyLoss`
  - **優化器**: `Adam`
  - **評估指標**: `Dice Score`, `IoU (Intersection over Union)`

---

## **Lab 5: 胸部 X 光疾病偵測**

### **目標**
- 使用 **Faster R-CNN** 進行 **胸部 X 光影像** 物件偵測。

### **技術與方法**
- **模型架構**:
  - **Faster R-CNN**:
    - 使用 **COCO 預訓練權重**。
    - 偵測 **主動脈硬鈣化、肺野浸潤、心臟肥大** 等多種疾病。
- **數據處理**:
  - **DICOM 影像轉換** 為標準化格式。
  - **對比度增強** (`Log Transformation`, `Simplest Color Balance`)。
- **訓練與評估**:
  - **損失函數**: `IoU`, `mAP`
  - **優化器**: `SGD with Momentum and Nesterov`
  - **評估指標**: `AP (Average Precision)`, `Recall`
- **可視化**:
  - 使用 **EigenCAM** 與 **AblationCAM** 分析模型關注區域。

---