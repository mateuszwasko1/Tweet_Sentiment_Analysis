from project_name.preprocessing.bert_preprocessing import (
    MainPreprocessing)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig)
from sklearn.metrics import (
    accuracy_score, classification_report, roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import joblib
import torch


class BertModel:
    def __init__(self):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def _tokenization(self, X):
        return self._tokenizer(
            X.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

    def _organize_data(self, data, batch_size=16):
        (X_training, y_training), (X_dev, y_dev), (X_test, y_test) = data

        number_of_labels = len(np.unique(y_training))

        X_training = self._tokenization(X_training)
        X_dev = self._tokenization(X_dev)
        X_test = self._tokenization(X_test)

        y_training = torch.tensor(y_training, dtype=torch.long)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_training.numpy()),
            y=y_training.numpy())
        self._class_weights = torch.tensor(
            weights,
            dtype=torch.float).to(self._device)

        y_dev = torch.tensor(y_dev, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(
            X_training["input_ids"], X_training["attention_mask"], y_training)
        dev_dataset = TensorDataset(
            X_dev["input_ids"], X_dev["attention_mask"], y_dev)
        test_dataset = TensorDataset(
            X_test["input_ids"], X_test["attention_mask"], y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, dev_loader, test_loader, number_of_labels

    def _get_model(self, number_of_labels, model_name, dropout):
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=number_of_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config)
        model.to(self._device)
        return model

    def _model_training(self,
                        model,
                        X_training,
                        X_dev,
                        epochs=15,
                        early_stopping=True,
                        lr=5e-5,
                        weight_decay=0.00):
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        EPOCHS = epochs

        if early_stopping:
            best_val_loss = 0
            patience = 3
            counter = 0
            best_model_state = None

        model.to(self._device)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            train_correct = 0
            train_total = 0

            for input_ids, attention_mask, labels in tqdm(
                    X_training,
                    desc=f"Epoch {epoch+1}/{EPOCHS}"):

                input_ids = input_ids.to(self._device)
                attention_mask = attention_mask.to(self._device)
                labels = labels.to(self._device)

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None)
                logits = output.logits

                loss_factor = CrossEntropyLoss(weight=self._class_weights)
                loss = loss_factor(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                predictions = logits.argmax(dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

            print(f"Epoch {epoch+1}: train loss =\
                  {total_loss/len(X_training):.4f}")
            print(f"Epoch {epoch+1}: train accuracy =\
                  {((train_correct/train_total)*100):.2f}%")

            model.eval()
            validation_loss = 0
            correct, total = 0, 0
            with torch.no_grad():
                for input_ids, attention_mask, labels in X_dev:

                    input_ids = input_ids.to(self._device)
                    attention_mask = attention_mask.to(self._device)
                    labels = labels.to(self._device)

                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
                    logits = output.logits
                    loss = output.loss

                    validation_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_loss = validation_loss/len(X_dev)
            print(f"Epoch {epoch+1}: dev accuracy =\
                  {(correct/total*100):.2f}%")
            print(f"Epoch {epoch+1}: val loss =\
                  {(val_loss):.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early Stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model

    def _saving_model(
            self,
            model,
            label_encoder,
            directory="data/model/saved_bert/"):
        model.save_pretrained(f"{directory}model")
        self._tokenizer.save_pretrained(f"{directory}model")
        joblib.dump(label_encoder, f"{directory}label_encoder")

    def _plot_roc_curve(self, y_true, y_score):
        classes = np.unique(y_true)
        y_bin = label_binarize(y_true, classes=classes)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i, class_id in enumerate(classes):
            fpr[class_id], tpr[class_id], _ = roc_curve(
                y_bin[:, i], y_score[:, i])
            roc_auc[class_id] = auc(fpr[class_id], tpr[class_id])

        plt.figure(figsize=(10, 7))
        for class_id in classes:
            class_label = self._label_encoder.inverse_transform([class_id])[0]
            plt.plot(fpr[class_id], tpr[class_id], label=(
                f"Class {class_label} (AUC = {roc_auc[class_id]:.2f})"))
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.grid(True)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for BERT Model")
        plt.legend(loc="lower right")
        plt.show()

    def _evaluation(self, model, X_test):
        model.eval()
        all_predictions = []
        all_lables = []
        all_probs = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in X_test:

                input_ids = input_ids.to(self._device)
                attention_mask = attention_mask.to(self._device)
                labels = labels.to(self._device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels).logits
                
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_predictions.extend(preds.cpu().tolist())
                all_lables.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().numpy())

        print("Test Accuracy:", accuracy_score(all_lables, all_predictions))
        print(classification_report(all_lables, all_predictions))
        self._plot_roc_curve(np.array(all_lables), np.array(all_probs))

    def pipeline(self):
        # Model Parameters #
        model_type = "roberta-base"
        early_stopping = True
        epochs = 15
        lr = 2e-5
        weight_decay = 0.01
        batch_size = 32
        dropout = 0.03
        save_directory = "models/saved_bert/"

        # Pipeline #
        ekphrasis_preprocessing = MainPreprocessing()
        data = ekphrasis_preprocessing.preprocessing_pipeline()
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            use_fast=False)
        train_loader, dev_loader, test_loader, number_of_labels = (
            self._organize_data(data, batch_size=batch_size))
        model = self._get_model(number_of_labels, model_type, dropout)
        best_model = self._model_training(
            model,
            train_loader,
            dev_loader,
            epochs=epochs,
            early_stopping=early_stopping,
            lr=lr,
            weight_decay=weight_decay)
        self._label_encoder = ekphrasis_preprocessing._label_encoder
        self._saving_model(
            best_model,
            self._label_encoder,
            directory=save_directory)
        metrics = self._evaluation(best_model, test_loader)
        return metrics
