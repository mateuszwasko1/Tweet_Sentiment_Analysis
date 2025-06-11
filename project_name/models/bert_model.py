import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from project_name.preprocessing.bert_preprocessing import MainPreprocessing

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))


class BertModel:
    """
    Fine-tunes and evaluates a BERT-based sequence classification model
    for emotion detection.
    """

    def __init__(self) -> None:
        """
        Initialize device (GPU if available, otherwise CPU).
        """
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._tokenizer: AutoTokenizer
        self._class_weights: Tensor
        self._label_encoder: np.ndarray

    def _tokenization(self, texts: np.ndarray) -> dict[str, Tensor]:
        """
        Tokenize a sequence of texts into input IDs and attention masks.

        Args:
            texts: Array of raw text strings.

        Returns:
            A dict containing 'input_ids' and 'attention_mask' tensors.
        """
        return self._tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

    def _organize_data(
        self,
        data: tuple[
            tuple[np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray],
        ],
        batch_size: int = 16,
    ) -> tuple[DataLoader, DataLoader, DataLoader, int]:
        """
        Convert raw features into DataLoaders and compute class weights.

        Args:
            data: ((X_train, y_train),
                   (X_dev,   y_dev),
                   (X_test,  y_test))
            batch_size: Number of samples per batch.

        Returns:
            Train, dev, and test DataLoaders, and number of unique labels.
        """
        (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = data

        num_labels = int(np.unique(y_train).size)

        X_train_tok = self._tokenization(X_train)
        X_dev_tok = self._tokenization(X_dev)
        X_test_tok = self._tokenization(X_test)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        self._class_weights = torch.tensor(
            weights, dtype=torch.float, device=self._device
        )

        y_dev_tensor = torch.tensor(y_dev, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_ds = TensorDataset(
            X_train_tok["input_ids"],
            X_train_tok["attention_mask"],
            y_train_tensor,
        )
        dev_ds = TensorDataset(
            X_dev_tok["input_ids"],
            X_dev_tok["attention_mask"],
            y_dev_tensor,
        )
        test_ds = TensorDataset(
            X_test_tok["input_ids"],
            X_test_tok["attention_mask"],
            y_test_tensor,
        )

        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(dev_ds, batch_size=batch_size),
            DataLoader(test_ds, batch_size=batch_size),
            num_labels,
        )

    def _get_model(
        self,
        num_labels: int,
        model_name: str,
        dropout: float,
    ) -> AutoModelForSequenceClassification:
        """
        Load a pretrained transformer with modified classification head.

        Args:
            num_labels: Number of output classes.
            model_name: HuggingFace model identifier.
            dropout: Dropout rate for classifier head.

        Returns:
            A model ready for fine-tuning.
        """
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        return model.to(self._device)

    def _model_training(
        self,
        model: AutoModelForSequenceClassification,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        epochs: int = 15,
        early_stopping: bool = True,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
    ) -> AutoModelForSequenceClassification:
        """
        Fine-tune the model on the training set, with optional early stopping.

        Args:
            model: Sequence classification model.
            train_loader: Training data loader.
            dev_loader:   Development data loader.
            epochs:       Maximum epochs to train.
            early_stopping: Whether to stop early on no improvement.
            lr:           Learning rate.
            weight_decay: Weight decay (L2) factor.

        Returns:
            The best model state (by lowest dev loss).
        """
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_val_loss = float("inf")
        patience, counter = 3, 0
        best_state = None

        model.to(self._device)

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            correct, total = 0, 0

            for input_ids, attention_mask, labels in tqdm(
                train_loader, desc=f"Epoch {epoch}/{epochs}"
            ):
                input_ids = input_ids.to(self._device)
                attention_mask = attention_mask.to(self._device)
                labels = labels.to(self._device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

                loss = CrossEntropyLoss(
                    weight=self._class_weights
                )(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = total_loss / len(train_loader)
            train_acc = correct / total * 100
            print(
                f"Epoch {epoch}: "
                f"train loss = {avg_train_loss:.4f}, "
                f"train acc = {train_acc:.2f}%"
            )

            # Validation
            model.eval()
            val_loss = 0
            val_correct, val_total = 0, 0

            with torch.no_grad():
                for input_ids, attention_mask, labels in dev_loader:
                    input_ids = input_ids.to(self._device)
                    attention_mask = attention_mask.to(self._device)
                    labels = labels.to(self._device)

                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    val_loss += output.loss.item()
                    preds = output.logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(dev_loader)
            val_acc = val_correct / val_total * 100
            print(
                f"Epoch {epoch}: "
                f"dev loss = {avg_val_loss:.4f}, "
                f"dev acc = {val_acc:.2f}%"
            )

            if early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    def _saving_model(
        self,
        model: AutoModelForSequenceClassification,
        label_encoder: np.ndarray,
        directory: str = "data/model/saved_bert/",
    ) -> None:
        """
        Persist the fine-tuned model and tokenizer to disk, plus label encoder.

        Args:
            model:          Fine-tuned transformer.
            label_encoder:  numpy array mapping indices to labels.
            directory:      Output directory prefix.
        """
        os.makedirs(directory, exist_ok=True)
        model.save_pretrained(f"{directory}model")
        self._tokenizer.save_pretrained(f"{directory}model")
        joblib.dump(label_encoder, f"{directory}label_encoder")

    def _plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> None:
        """
        Plot ROC curves for each class.

        Args:
            y_true:  Ground-truth labels.
            y_score: Predicted class probabilities.
        """
        classes = np.unique(y_true)
        y_bin = label_binarize(y_true, classes=classes)

        plt.figure(figsize=(10, 7))
        for i, class_id in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            class_label = self._label_encoder[class_id]
            plt.plot(
                fpr,
                tpr,
                label=(
                    f"Class {class_label} "
                    f"(AUC = {roc_auc:.2f})"
                ),
            )

        plt.plot(
            [0, 1],
            [0, 1],
            "k--",
            label="Random",
        )
        plt.grid(True)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for BERT Model")
        plt.legend(loc="lower right")
        plt.show()

    def _evaluation(
        self,
        model: AutoModelForSequenceClassification,
        test_loader: DataLoader,
    ) -> None:
        """
        Evaluate model on the test set: accuracy, classification report,
        and ROC curves.

        Args:
            model:       Fine-tuned transformer.
            test_loader: DataLoader for test data.
        """
        model.eval()
        preds, labels, probs = [], [], []

        with torch.no_grad():
            for input_ids, attention_mask, lbls in test_loader:
                input_ids = input_ids.to(self._device)
                attention_mask = attention_mask.to(self._device)
                lbls = lbls.to(self._device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

                prob = torch.nn.functional.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)

                preds.extend(pred.cpu().tolist())
                labels.extend(lbls.cpu().tolist())
                probs.extend(prob.cpu().numpy())

        print(
            "Test Accuracy:",
            accuracy_score(labels, preds),
        )
        print(classification_report(labels, preds))
        self._plot_roc_curve(
            np.array(labels),
            np.array(probs),
        )

    def pipeline(self) -> None:
        """
        End-to-end: preprocess data, fine-tune model, save artifacts,
        and evaluate on test set.
        """
        # Hyperparameters
        model_name = "roberta-base"
        early_stopping = True
        epochs = 15
        lr = 2e-5
        weight_decay = 0.01
        batch_size = 32
        dropout = 0.03
        save_dir = "models/saved_bert/"

        # Preprocess
        preproc = MainPreprocessing()
        data = preproc.preprocessing_pipeline()

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )
        train_dl, dev_dl, test_dl, num_labels = self._organize_data(
            data,
            batch_size=batch_size,
        )

        model = self._get_model(
            num_labels,
            model_name,
            dropout,
        )
        best_model = self._model_training(
            model,
            train_dl,
            dev_dl,
            epochs=epochs,
            early_stopping=early_stopping,
            lr=lr,
            weight_decay=weight_decay,
        )

        # Persist artifacts
        self._label_encoder = preproc._label_encoder
        self._saving_model(
            best_model,
            self._label_encoder,
            directory=save_dir,
        )

        self._evaluation(best_model, test_dl)
