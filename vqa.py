import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Dataset Loading and Preprocessing
class VQADataset(Dataset):
    def __init__(self, image_dir, question_json, answer_json, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Load questions and answers
        with open(question_json, 'r') as f:
            self.questions = json.load(f)['questions']
        
        with open(answer_json, 'r') as f:
            self.answers = json.load(f)['annotations']
        
        # Tokenize questions
        self.tokenize_questions()
        
        # Create answer vocabulary
        self.create_answer_vocab()
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        img_id = question['image_id']
        question_id = question['question_id']
        image_path = os.path.join(self.image_dir, 'COCO_train2014_' + '%012d.jpg' % img_id)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Get question
        question = question['question']
        question_tensor = torch.tensor(self.encode_question(question))
        
        # Get answer
        answer = self.answers[idx]['multiple_choice_answer']
        answer_idx = self.answer_vocab[answer]
        answer_tensor = torch.tensor(answer_idx)
        
        return image, question_tensor, answer_tensor
    
    def tokenize_questions(self):
        for question in self.questions:
            question['question'] = word_tokenize(question['question'].lower())
            
    def create_answer_vocab(self):
        answers = [item['multiple_choice_answer'] for item in self.answers]
        answer_counts = Counter(answers)
        self.answer_vocab = {answer: idx for idx, (answer, _) in enumerate(answer_counts.items())}
        
    def encode_question(self, question):
        return [self.vocab.get(token, self.vocab['<unk>']) for token in question]

#  Data Loading and Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define paths to the dataset files
image_dir = '/path/to/COCO/images/train2014'
question_json = '/path/to/question_file.json'
answer_json = '/path/to/answer_file.json'

# Initialize dataset and dataloader
dataset = VQADataset(image_dir, question_json, answer_json, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


import torch.nn as nn
import torchvision.models as models

# Usage Example in a Model (Sample)
class VQAModel(nn.Module):
    def __init__(self, num_classes):
        super(VQAModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1024)
        self.question_rnn = nn.LSTM(input_size=300, hidden_size=512, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, image, question):
        image_features = self.resnet(image)
        question_features, _ = self.question_rnn(question)
        combined_features = torch.cat((image_features, question_features[:, -1, :]), dim=1)
        x = self.dropout(F.relu(self.fc1(combined_features)))
        x = self.fc2(x)
        return x

# Initialize model and optimizer
model = VQAModel(num_classes=len(dataset.answer_vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (sample)
for epoch in range(num_epochs):
    for batch_idx, (images, questions, answers) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images, questions)
        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()
        
        if (batch_idx+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
