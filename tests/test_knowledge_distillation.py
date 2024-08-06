import torch
import torch.nn as nn
import unittest
from solv_ai import DistillationLoss

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

class TestKnowledgeDistillation(unittest.TestCase):
    def test_distillation_loss(self):
        teacher_model = TeacherModel()
        student_model = StudentModel()
        distillation_loss = DistillationLoss(teacher_model, student_model)

        x = torch.randn(3, 10)
        target = torch.randint(0, 5, (3,))
        loss = distillation_loss(x, target)
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()