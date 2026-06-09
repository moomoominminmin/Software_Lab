import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim

# define MyLeNet
class MyLet5_1(nn.Module) :
    # CNN 부품들을 생성하는 부분 (conv, maxpool, relu, linear)
    def __init__(self):
        super(MyLet5_1, self).__init__() # 부모 클래스(nn.Module의 __init__)을 실행함, 현재 객체가 부모크래스의 __init__을 가져다 사용
        self.conv_1 = nn.Conv2d(1,6, kernel_size = 5, padding= 2)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv_2 = nn.Conv2d(6,16,kernel_size=5)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv_3 = nn.Conv2d(16,120, kernel_size=5)
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(120, 84)
        self.fc_2 = nn.Linear(84,10)

    # 전의 만든 부품들을 이용해서 모델을 구축하는 부분 (conv->relu->conv->relu...)
    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 120)
        x = self.fc_1(x)
        x = self.relu(x)
        res = self.fc_2(x)
        return res

model = MyLet5_1()
print(model)

# Model creation & set up hyper parameters
# 모델 파라미터를 설정하는 부분 (batch, learning rate, epoch)
# 모델을 만들고 학습 환경을 설정하는 코드
batch_size = 16 # 한 번에 몇 장의 이미지를 학습할 것인지
learning_rate = 0.01 
epochs = 5 # 전체 데이터셋을 몇 번 반복할 것인가
#use_cuda = True
#device = torch.device("cuda" if use_cuda else "cpu")
# 모델 구조를 만드는 데 필요한 코드가 아니라 학습 속도를 높이기 위한 코드
use_cuda = torch.cuda.is_available() # GPU가 있는지 검사
device = torch.device("cuda" if use_cuda else "cpu") 

model = MyLet5_1() # 객체(인스턴스) 생성
model.to(device) # 모델을 CPU에 둘지 CPU에 둘지 결정

#print('device: ', device)
# or model.to("cuda:0")
#print(next(model.parameters()).is_cuda)

# Dataload
# train data와 test data를 load함
# GPU 사용 시 데이터 로딩 최적화 옵션
kwargs = { 'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_dataset = datasets.MNIST(
    './data', train = True, download=True,
    transform=transforms.Compose([ #Compose() 전처리 여러 개를 순서대로 실행
                                transforms.ToTensor(), # 0~1 범위로 변환
                                transforms.Normalize((0.1307,), (0.3081,)) # 데이터 분포를 정규화
                                ])
    )
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

test_dataset = datasets.MNIST(
    './data', train = False,
    transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])
    )
train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

print('[Trainset: ]', train_dataset)
print('[Testset: ]', test_dataset)

# Optimization
opimtizer = optim.Adadelta(model.parameters(), lr=learning_rate) # 가중치 최적화, adadelta라는 최적화 함수 사용, adadelta(최적화해야할 가중치 개수, 학습률)
scheduler = optim.lr_scheduler.StepLR(opimtizer, step_size=1, gamma=0.1) # learning rate 조절 함수
loss_function = nn.CrossEntropyLoss() # cross entropy loss cost function을 사용

# define train
def train(model, device, train_loader, optimizer, loss_function, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        opimtizer.step()

        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))


# define test
def test(model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    test_loss /= len(test_loader.dataset)
    print(    )

# run test
for epoch in range(epochs):
    train(model, device, train_loader, opimtizer, loss_function, epoch)
    test(model, device, test_loader, loss_function)
    scheduler.step()
