#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// 학습에 사용될 입력데이터와 라벨값을 벡터 형태로 선언 및 초기화
vector<double> x_data = {2.0, 4.0, 6.0, 8.0};
vector<double> y_data = {4.0, 8.0, 12.0, 16.0};

double forward(double x, double w);
double loss(double x, double y, double w);

int main(){
    // 다양한 가중치와 그에 따른 MSE를 저장할 리스트 생성
    vector<double> w_list;
    vector<double> mse_list;

    // 가중치를 0부터 4까지 0.1 씩 증가하면서 탐색함
    for (double w = 0.0; w <= 4.0; w+= 0.1){
        cout << "w=" << w << endl;
        double l_sum = 0.0; // 오차값 초기화

        // 전체 데이터를 돌며 개별 오차를 계산하는 내부 반복문
        for (size_t i = 0; i < x_data.size(); i++){
            double x_val = x_data[i];
            double y_val = y_data[i];
            double y_pred_val = forward(x_val, w);
            double l = loss(x_val, y_val, w);
            l_sum += l;
            cout << "\t" << x_val << " " << y_val << " " << y_pred_val << " " << l << endl;
        }
        cout << "MSE =" << l_sum / x_data.size() << endl;
        w_list.push_back(w);
        mse_list.push_back(l_sum/x_data.size());
        return 0;
    }
}

double forward(double x, double w){
    return x*w;
}

double loss(double x, double y, double w){
    double y_pred = forward(x, w);
    return pow(y_pred - y, 2);
}

