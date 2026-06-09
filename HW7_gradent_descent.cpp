#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// 학습에 사용될 입력데이터와 라벨값을 벡터 형태로 선언 및 초기화
vector<double> x_data = {2.0, 4.0, 6.0, 8.0};
vector<double> y_data = {4.0, 8.0, 12.0, 16.0};

double w = 1.0;

double forward(double x);
double loss(double x, double y);
double gradient(double x, double y);

int main(){
    cout << "Prediction (before learning) : " << forward(10) << endl;

    for (int epoch = 0; epoch < 10; epoch++){
        double loss_sum = 0;
        for (size_t i = 0; i < x_data.size(); i++){
            double x_val = x_data[i];
            double y_val = y_data[i];

            double grad = gradient(x_val, y_val);
            w = w - 0.01 * grad;

            cout << "\tgrad: " << x_val << " " << y_val << " " << round(grad*100)/100 << endl;
            loss_sum += loss(x_val, y_val);
        }
        cout << "progress: " << epoch << "w = " << round(w*100)/100 << " loss_sum = " << round(loss_sum * 100)/100 << endl;
    }
    cout << "Predicted score (after training) :"  << "10 hours of working" << forward(10)<< endl;
    return 0;
}

double forward(double x){
    return x*w;
}

double loss(double x, double y){
    double y_pred = forward(x);
    return pow(y_pred - y, 2);
}

double gradient(double x, double y) {
    return 2 * x * (x * w - y);
}
