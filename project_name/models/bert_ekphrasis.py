from project_name.preprocessing.ekphrasis_preprocessing import MainPreprocessing
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib


class BertModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

    def tokenization(self, X):
        train_encodings = self.tokenizer(
            X.tolist(),
            truncation = True,
            padding = True,
            max_length = 128,
            return_tensors = "pt"
        )
        return train_encodings

    def organize_data(self, data):
        (X_training, y_training), (X_dev, y_dev), (X_test, y_test) = data

        # train_indices = X_training.sample(n=100, random_state=42).index
        # X_training = X_training.loc[train_indices]
        # y_training = y_training[train_indices]
        
        # dev_indices = X_dev.sample(n=30, random_state=42).index
        # X_dev = X_dev.loc[dev_indices]
        # y_dev = y_dev[dev_indices]
        
        # test_indices = X_test.sample(n=30, random_state=42).index
        # X_test = X_test.loc[test_indices]
        # y_test = y_test[test_indices]

        number_of_lables = len(np.unique(y_training))

        X_training = self.tokenization(X_training)
        X_dev = self.tokenization(X_dev)
        X_test = self.tokenization(X_test)

        y_training = torch.tensor(y_training, dtype=torch.long)
        y_dev = torch.tensor(y_dev, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        X_training = TensorDataset(X_training["input_ids"], X_training["attention_mask"], y_training)
        X_dev = TensorDataset(X_dev["input_ids"], X_dev["attention_mask"], y_dev)
        X_test = TensorDataset(X_test["input_ids"], X_test["attention_mask"], y_test)

        batch_size = 32
        X_training = DataLoader(X_training, batch_size=batch_size, shuffle=True)
        X_dev = DataLoader(X_dev, batch_size=batch_size)
        X_test = DataLoader(X_test, batch_size=batch_size)

        return X_training, X_dev, X_test, number_of_lables #CHANGE NAMES

    def get_model(self, number_of_labels):
        model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=number_of_labels)
        return model
        
    def model(self, model, X_training, X_dev, X_test, early_stopping=True):
        optimizer = AdamW(model.parameters(), lr=2e-5)
        EPOCHS = 15
        
        if early_stopping:
            best_val_accuracy = 0
            patience = 2
            counter = 0
            best_model_state = None

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            for input_ids, attention_mask, labels in tqdm(X_training, desc=f"Epoch {epoch+1}/{EPOCHS}"):

                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: train loss = {total_loss/len(X_training):.4f}")

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for input_ids, attention_mask, labels in X_dev:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_accuracy = correct/total
            print(f"Epoch {epoch+1}: dev accuracy = {correct/total:.2%}")            

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter +=1
                if counter >= patience:
                    print(f"Early Stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model
    
    def saving_model(self, model, label_encoder): 
        model.save_pretrained("data/model/saved_bert/model")
        self.tokenizer.save_pretrained("data/model/saved_bert/model")
        joblib.dump(label_encoder, "data/model/saved_bert/label_encoder")
        


    def evaluation(self, model, X_train):
        model.eval()
        all_predictions = []
        all_lables = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in X_train:
                logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
                preds = logits.argmax(dim=1)
                all_predictions.extend(preds.tolist())
                all_lables.extend(labels.tolist())

        print("Test Accuracy:", accuracy_score(all_lables, all_predictions))
        print(classification_report(all_lables, all_predictions))

    def pipeline(self):
        ekphrasis_preprocessing = MainPreprocessing()
        data = ekphrasis_preprocessing.preprocessing_pipeline()
        X_training, X_dev, X_test, number_of_lables  = self.organize_data(data)
        model = self.get_model(number_of_lables)
        best_model = self.model(model, X_training, X_dev, X_test)
        label_encoder = ekphrasis_preprocessing.label_encoder
        self.saving_model(best_model, label_encoder)
        metrics = self.evaluation(best_model, X_training)
        return metrics

