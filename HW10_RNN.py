import torch # PyTorch 라이브러리 
import torch.nn as nn # 신경망 모듈 
import torch.optim as optim # 최적화 함수
import numpy as np # NumPy 라이브러

chars = "abcdefghijklmnopqrstuvwxyz" # 알파벳 소문자 문자열 정의
char_list = [i for i in chars] # 문자열을 문자 리스트로 변환
n_letters = len(char_list) # 알파벳 개수 (26개)

n_layers = 1 # RNN 레이어 개수 설정

five_words = ['basic','beach','below','black','brown','carry','cream',
'drink','error','event','exist','first','funny','guess','human','image',
'large','magic','mouse','night','noise','ocean','often','order','peace',
'phone','print','quiet','reach','rough','round','scene','score','sense',
'skill','sleep','small','storm','table','think','touch','twice','until',
'upset','voice','waste','watch','white','woman','young'] # 학습에 사용할 50개의 단어 리스트 정의

n_five_words = len(five_words) # 50개 단어의 개수 저장

sequence_length = 4 # 시퀀스 길이 (단어의 길이 -1, 예측할 문자 수)

def word_to_onehot(string): # 단어를 원-핫 인코딩으로 변환하는 함수 정의
    one_hot = np.array([]).reshape(0, n_letters) # 비어있는 배열 초기화 (0행, n_letters열)

    for i in string: # 입력 문자열의 각 문자에 대해 반복
        idx = char_list.index(i) # 문자의 인덱스 찾기

        zero = np.zeros(shape=n_letters, dtype=int) # 모든 요소가 0인 배열 생성
        zero[idx] = 1 # 해당 문자의 인덱스에 1 설정 (one-hot-encdoing)

        one_hot = np.vstack([one_hot, zero]) # 생성된 원-핫 벡터를 전체 배열에 수직으로 쌓기

    return one_hot # 원-핫 인코딩된 배열 반환

def onehot_to_word(onehot_1): # 원-핫 벡터를 문자로 변환하는 함수 정의
    onehot = torch.Tensor.numpy(onehot_1) # PyTorch tensor를 NumPy 배열로 변환
    return char_list[onehot.argmax()] # 가장 큰 값의 인덱스에 해당하는 문자를 반환

class myRNN(nn.Module): # 커스텀 RNN 모델 클래스 정의 (nn.Module 상속)
    def __init__(self, input_size, hidden_size, num_layer): # 초기화 메서드
        super(myRNN, self).__init__() # nn.Module의 초기화 메서드 호출

        self.input_size = input_size # 입력 크기 저장 (알파벳 개수)
        self.hidden_size = hidden_size # 은닉 상태 크기 저장
        self.num_layer = num_layer # RNN 레이어 개수 저장

        self.rnn = nn.RNN( # PyTorch RNN 레이어 정의
            input_size=input_size, # 입력 크기
            hidden_size=hidden_size, # 은닉 상태 크기
            num_layers=num_layer # RNN 레이어 개수
        )

    def forward(self, x, hidden): # 순전파 메서드
        out, hidden = self.rnn(x, hidden) # RNN에 입력과 은닉 상태 전달하여 출력과 새 은닉 상태 얻기
        return out, hidden # 출력과 은닉 상태 반환

    def init_hidden(self): # 초기 은닉 상태를 생성하는 메서드
        return torch.zeros(self.num_layer, 1, self.hidden_size) # 모든 요소가 0인 은닉 상태 텐서 반환

def main(): # 메인 학습 및 평가 함수 정의
    n_hidden = 26 # 은닉 상태 크기 설정
    lr = 0.001 # 학습률 설정
    epochs = 900 # 학습 에폭(반복) 수 설정

    model = myRNN(n_letters, n_hidden, n_layers) # myRNN 모델 초기화

    loss_func = nn.CrossEntropyLoss() # 손실 함수로 교차 엔트로피 손실 사용

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam 옵티마이저 사용

    scheduler = optim.lr_scheduler.StepLR( # 학습률 스케줄러 정의
        optimizer, # 옵티마이저
        step_size=300, # 300 에폭마다
        gamma=0.1 # 학습률을 0.1배로 감소
    )

    for i in range(epochs): # 지정된 에폭 수만큼 반복
        total_loss = 0 # 각 에폭의 총 손실 초기화

        for j in range(n_five_words): # 50개의 단어 각각에 대해 반복
            hidden = model.init_hidden() # 각 단어 학습 시작 시 은닉 상태 초기화

            string = five_words[j] # 현재 단어 가져오기

            one_hot = torch.from_numpy( # 단어를 원-핫 인코딩으로 변환하고 PyTorch 텐서로 만들기
                word_to_onehot(string)
            ).type_as(torch.FloatTensor()) # 데이터 타입을 FloatTensor로 설정

            model.zero_grad() # 모델의 모든 학습 가능한 파라미터의 기울기 0으로 초기화

            hidden = model.init_hidden() # 다시 은닉 상태 초기화 (여기서 다시 초기화하는 이유는 RNN의 각 시퀀스마다 독립적인 은닉 상태를 사용하기 위함)

            input = one_hot[0:-1] # 입력 시퀀스: 첫 번째 글자부터 마지막 글자 직전까지
            input = torch.unsqueeze(input, 1) # 입력 텐서에 배치 차원 추가 (sequence_length, 1, input_size)

            target = torch.argmax(one_hot[1:], dim=1) # 목표 시퀀스: 두 번째 글자부터 마지막 글자까지 (인덱스 형태), PyTorch 텐서로 직접 계산

            output, hidden = model(input, hidden) # 모델에 입력 시퀀스와 은닉 상태를 전달하여 예측 수행

            loss = loss_func(output.squeeze(1), target) # 출력과 목표를 사용하여 손실 계산 (output.squeeze(1)로 배치 차원 제거)

            loss.backward() # 역전파를 통해 기울기 계산
            optimizer.step() # 옵티마이저를 사용하여 모델 파라미터 업데이트

        if i % 10 == 0: # 10 에폭마다 손실 출력
            print('epoch:%d' % i) # 현재 에폭 출력
            print(loss) # 현재 에폭의 손실 출력

        scheduler.step() # 학습률 스케줄러 업데이트

    torch.save(model.state_dict(), 'trained.pth') # 학습된 모델의 가중치를 'trained.pth' 파일로 저장

    model.load_state_dict(torch.load('trained.pth')) # 저장된 가중치를 모델에 로드 (다시 로드하여 평가 준비)

    with torch.no_grad(): # 기울기 계산을 비활성화 (평가 모드)
        total = 0 # 전체 단어 예측 수 초기화
        positive = 0 # 정확하게 예측된 단어 수 초기화

        total_text = 0 # 전체 문자 예측 수 초기화
        positive_text = 0 # 정확하게 예측된 문자 수 초기화

        for i in range(n_five_words): # 50개의 단어 각각에 대해 반복
            string = five_words[i] # 현재 단어 가져오기

            one_hot = torch.from_numpy( # 단어를 원-핫 인코딩으로 변환하고 PyTorch 텐서로 만들기
                word_to_onehot(string)
            ).type_as(torch.FloatTensor()) # 데이터 타입을 FloatTensor로 설정

            hidden = model.init_hidden() # 은닉 상태 초기화

            input = one_hot[0:-1] # 입력 시퀀스: 첫 번째 글자부터 마지막 글자 직전까지
            input = torch.unsqueeze(input, 1) # 입력 텐서에 배치 차원 추가

            target = torch.argmax(one_hot[1:], dim=1) # 목표 시퀀스 (평가에는 직접 사용되지 않지만 코드 일관성을 위해 유지)

            output, hidden = model(input, hidden) # 모델에 입력 시퀀스와 은닉 상태 전달하여 예측 수행

            output = output.squeeze() # 출력 텐서에서 배치 차원 제거

            output_string = string[0] # 예측 문자열의 첫 글자는 실제 단어의 첫 글자로 초기화

            for j in range(output.size()[0]): # 예측된 각 문자에 대해 반복
                output_string += onehot_to_word(output[j].data) # 예측된 원-핫 벡터를 문자로 변환하여 예측 문자열에 추가

                total_text += 1 # 전체 문자 예측 수 증가

                if string[j+1] == output_string[-1]: # 실제 단어의 다음 문자와 예측된 문자가 같으면
                    positive_text += 1 # 정확하게 예측된 문자 수 증가

            total += 1 # 전체 단어 예측 수 증가

            if string[-1] == output_string[-1]: # 실제 단어의 마지막 문자와 예측된 문자열의 마지막 문자가 같으면
                positive += 1 # 정확하게 예측된 단어 수 증가

            print('%d GT:%s OUT:%s' % (i+1, string, output_string)) # 각 단어의 예측 결과 출력 (Ground Truth: 실제 단어, Output: 예측 단어)

    print(
        'final text accuracy %d/%d (%.4f)' # 최종 단어 예측 정확도 출력
        % (positive, total, positive / total)
    )

    print(
        'whole text accuracy %d/%d (%.4f)' # 전체 문자 예측 정확도 출력
        % (
            positive_text,
            total_text,
            positive_text / total_text
        )
    )

if __name__ == '__main__': # 스크립트가 직접 실행될 때 main() 함수 호출
    main()
