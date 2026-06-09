#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// 학습에 사용될 입력데이터와 라벨값을 벡터 형태로 선언 및 초기화
vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
vector<double> y_data = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

// 초기 가중치와 bias
double w = 1.0;
double b = 0.0;

double sigmoid(double z);
double forward(double x);
double loss(double x, double y);
double gradient(double x, double y, double& grad_w, double& grad_b);

int main(){
    // 학습 전
    cout << "Prediction (before training) for x=1: " << forward(1) << endl;
    cout << "Prediction (before training) for x=9: " << forward(10) << endl;

    // Training Loop
    for (int epoch = 0; epoch < 200; epoch++){ // 전체 데이터를 학습하는 것 반복 횟수
        double loss_sum = 0.0;
        cout << "Epoch[" << epoch << "]" << endl;
        for (size_t i = 0; i < x_data.size(); i++){ // x 안에 있는 데이터를 돌면서 학습 진행
            // 현재 x와 y 값을 각각 x_val과 y_val에 넣기
            double x_val = x_data[i]; 
            double y_val = y_data[i];
            double grad_w, grad_b;

            gradient(x_val, y_val, grad_w, grad_b); // gradient 계산하여 가중치와 bias 값 수정
            w -= 0.01*grad_w; // 가중치 이동
            b -= 0.01*grad_b; // bias 이동
            cout << " \tgrad: " << " x= " << x_val << " y= " << y_val << " grad_w= " << grad_w << " grad_b= " << grad_b << endl;
            loss_sum += loss(x_val, y_val); // 손실값 누적
        }
        cout << "progress: " << epoch << "w = " << w << " b= " << b << " loss_sum = " << loss_sum << endl << endl;
    }
    cout << "Predicted score (after training) for x=1: " << forward(1) << endl;
    cout << "Predicted score (after training) for x=9: " << forward(9) << endl;
    return 0;
}

double sigmoid(double z){
    return 1 / (1 + exp(-z));
}

double forward(double x){
    double z = x*w + b;
    return sigmoid(z);
}

// Binary Cross Entropy Loss
double loss(double x, double y){
    double y_pred = forward(x);
    return -(y*log(y_pred)+(1-y)*log(1-y_pred));
}

double gradient(double x, double y, double& grad_w, double& grad_b) {
    double y_pred = forward(x);
    grad_w = (y_pred - y) * x;
    grad_b = (y_pred - y);
}
