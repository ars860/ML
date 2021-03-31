#include <iostream>
#include <utility>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <functional>

using namespace std;

pair<pair<vector<vector<double>>, vector<vector<double>>>, pair<vector<double>, vector<double>>>
splitOnDataAndTest(vector<vector<double>> &wholeDataset, vector<double> classes, int testsAmount, int testIndex);

void printVec(vector<double> &x) {
    for (double val : x) {
        cout << val << " ";
    }
    cout << '\n';
}

double scalar(vector<double> &x, vector<double> &y) {
    assert(x.size() == y.size());
    double res = 0;
    for (int i = 0; i < x.size(); i++) {
        res += x[i] * y[i];
    }

    return res;
}

vector<double> sub(vector<double> &x, vector<double> &y) {
    assert(x.size() == y.size());
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = x[i] - y[i];
    }

    return res;
}

double euclideanDistanceKvadrat(vector<double> &x, vector<double> &y) {
    assert(x.size() == y.size());
    int n = x.size();
    double dist = 0;
    for (int i = 0; i < n; i++) {
        dist += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return dist;
}

function<double(vector<double>, vector<double>)> makeLinearKernel(double c) {
    return [=](vector<double> p1, vector<double> p2) -> double {
        return scalar(p1, p2) + c;
    };
}

function<double(vector<double>, vector<double>)> makePolynomialKernel(double alpha, double c, double d_) {
    return [=](vector<double> p1, vector<double> p2) -> double {
        return pow(alpha * scalar(p1, p2) + c, d_);
    };
}

function<double(vector<double>, vector<double>)> makeExponentialKernel(double beta) {
    return [=](vector<double> p1, vector<double> p2) -> double {
        double dist = euclideanDistanceKvadrat(p1, p2);

        return exp(-beta * dist);
    };
}

vector<double> a;
vector<double> target;
vector<double> error_cache;
vector<vector<double>> d;
function<double(vector<double> x, vector<double> y)> kernel;
double C = 1;
double eps = 0.001;
double b = 0;
double tol = 0.001;

bool precalc = true;
vector<vector<double>> precalced;

double kernelBetweenRowsInDataset(int i1, int i2) {
    if (precalc) {
        return precalced[i1][i2];
    }

    return kernel(d[i1], d[i2]);
}

double calcPrediction(int index) {
//    assert(d.size() == d[0].size());
    assert(d.size() == a.size());
    assert(d.size() == target.size());

    double ans = 0;

//    calcB();
    for (int i = 0; i < d.size(); i++) {
        if (a[i] > 0) {
            ans += a[i] * target[i] * kernelBetweenRowsInDataset(index, i);//d[index][i];
        }
    }

    ans -= b;

    return ans;
}

int takeStep(int i1, int i2) {
    if (i1 == i2) {
        return 0;
    }

    double alph1 = a[i1];
    double alph2 = a[i2];
    double y1 = target[i1];
    double y2 = target[i2];
    double s = y1 * y2;

    double E1 = alph1 > 0 && alph1 < C ? error_cache[i1] : calcPrediction(i1) - y1;
    double E2 = alph2 > 0 && alph2 < C ? error_cache[i2] : calcPrediction(i2) - y2;

    double L, H;
    if (y1 == y2) {
        if (alph1 + alph2 > C) {
            L = alph1 + alph2 - C;
            H = C;
        } else {
            L = 0;
            H = alph1 + alph2;
        }
    } else {
        if (alph1 - alph2 > 0) {
            L = 0;
            H = C - (alph1 - alph2);
        } else {
            L = alph2 - alph1;
            H = C;
        }
    }

    if (L == H) {
        return 0;
    }

    double k11 = kernelBetweenRowsInDataset(i1, i1);// d[i1][i1];
    double k12 = kernelBetweenRowsInDataset(i1, i2);// d[i1][i2];
    double k22 = kernelBetweenRowsInDataset(i2, i2);// d[i2][i2];
    double eta = 2 * k12 - k11 - k22;

    double a2;
    if (eta < 0) {
        a2 = alph2 + y2 * (E2 - E1) / eta;
        if (a2 < L) {
            a2 = L;
        } else {
            if (a2 > H) {
                a2 = H;
            }
        }
    } else {
//        calcB();

        double c1 = eta / 2;
        double c2 = y2 * (E1 - E2) - eta * alph2;
        double Lobj = c1 * L * L + c2 * L;
        double Hobj = c1 * H * H + c2 * H;

        if (Lobj > Hobj + eps) {
            a2 = L;
        } else {
            if (Lobj < Hobj - eps) {
                a2 = H;
            } else {
                a2 = alph2;
            }
        }
    }

    if (abs(a2 - alph2) < eps * (a2 + alph2 + eps)) {
        return 0;
    }

    double a1 = alph1 + s * (alph2 - a2);

    if (a1 < 0) {
        a2 += s * a1;
        a1 = 0;
    } else if (a1 > C) {
        a2 += s * (a1 - C);
        a1 = C;
    }


//    Update threshold to reflect change in Lagrange multipliers
//    b = -1;

    double bnew;
    if (a1 > 0 && a1 < C) {
        bnew = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
    } else {
        if (a2 > 0 && a2 < C) {
            bnew = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
        } else {
            double b1 = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
            double b2 = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;

            bnew = (b1 + b2) / 2;
        }
    }
    double delta_b = bnew - b;
    b = bnew;

//    Update weight vector to reflect change in a1 & a2, if SVM is linear ???

//    Update error cache using new Lagrange multipliers
    for (int i = 0; i < d.size(); i++) {
        if (0 < a[i] && a[i] < C) {
            error_cache[i] +=
                    y1 * (a1 - alph1) * kernelBetweenRowsInDataset(i1, i) + //d[i1][i] +
                    y2 * (a2 - alph2) * kernelBetweenRowsInDataset(i2, i) - //d[i2][i] -
                    delta_b;
        }
    }

    error_cache[i1] = 0;
    error_cache[i2] = 0;

    a[i1] = a1;
    a[i2] = a2;

    return 1;
}

int examineExample(int i1) {
    double y1 = target[i1];
    double alph1 = a[i1];

    double E1;
    if (alph1 > 0 && alph1 < C) {
        E1 = error_cache[i1];
    } else {
        E1 = calcPrediction(i1) - y1;
    }

    double r1 = y1 * E1;

    if ((r1 < -tol && alph1 < C) || (r1 > tol && alph1 > 0)) {
        pair<double, int> maxDiff = {-1, -1};

        for (int i = 0; i < a.size(); i++) {
            if (a[i] > 0 && a[i] < C) {
                maxDiff = max(maxDiff, {abs(E1 - error_cache[i]), i});
            }
        }

        if (maxDiff.second != -1) {
            if (takeStep(i1, maxDiff.second))
                return 1;
        }

        for (int i = 0; i < a.size(); i++) {
            if (a[i] > 0 && a[i] < C) {
                if (takeStep(i1, i))
                    return 1;
            }
        }

        for (int i = 0; i < a.size(); i++) {
            if (takeStep(i1, i)) {
                return 1;
            }
        }

    }

    return 0;
}

void initGlobals(double C_, function<double(vector<double>, vector<double>)> _kernel) {
    int n = d.size();

    C = C_;
    b = 0;
    error_cache = vector<double>(n);
    a = vector<double>(n);
    kernel = std::move(_kernel);

    if (precalc) {
        precalced = vector<vector<double>>(n, vector<double>(n));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                precalced[i][j] = kernel(d[i], d[j]);
            }
        }
    }
}

int maxIterations = 1000;

vector<double> doShit() {
    int iter = 0;
    int n = d.size();

    int numChanged = 0;
    bool examineAll = true;

    while (numChanged > 0 || examineAll) {
        if (iter++ > maxIterations) {
            break;
        }

        numChanged = 0;
        if (examineAll) {
            for (int i = 0; i < n; i++) {
                numChanged += examineExample(i);
            }
        } else {
            for (int i = 0; i < n; i++) {
                if (a[i] != 0 && a[i] != C) {
                    numChanged += examineExample(i);
                }
            }
        }

        if (examineAll) {
            examineAll = false;
        } else {
            if (numChanged == 0) {
                examineAll = true;
            }
        }
    }

    return a;
}

//double calcPrediction(int index) {
////    assert(d.size() == d[0].size());
//    assert(d.size() == a.size());
//    assert(d.size() == target.size());
//
//    double ans = 0;
//
////    calcB();
//    for (int i = 0; i < d.size(); i++) {
//        if (a[i] > 0) {
//            ans += a[i] * target[i] * kernel(d[index], d[i]);//d[index][i];
//        }
//    }
//
//    ans -= b;
//
//    return ans;
//}

bool classificator(vector<double> &point) {
    double regr = 0;
    for (int i = 0; i < d.size(); i++) {
        regr += a[i] * target[i] * kernel(point, d[i]);
    }
    regr -= b;

    return regr > 0;
}

double calcAccuracy(vector<vector<double>> &test, vector<double> &test_classes) {
    double classifiedCorrectly = 0;
    for (int i = 0; i < test.size(); i++) {
        bool isOne = classificator(test[i]);

        if ((isOne && test_classes[i] == 1) || (!isOne && test_classes[i] == -1)) {
            classifiedCorrectly++;
        }
    }

    return classifiedCorrectly / test.size();
}

void cinParameters() {
    string name;
    cin >> C;
    cin >> name;

    if (name == "linear") {
        double c;
        cin >> c;

        initGlobals(C, makeLinearKernel(c));
        return;
    }
    if (name == "polynomial") {
        double alpha, c;
        int d_;
        cin >> c >> d_ >> alpha;

        initGlobals(C, makePolynomialKernel(alpha, c, d_));
        return;
    }
    if (name == "exponential") {
        double beta;
        cin >> beta;

        initGlobals(C, makeExponentialKernel(beta));
        return;
    }
    throw runtime_error("unknown kernel: " + name);
}

void learnLinear(vector<vector<double>> &wholeDataSet, vector<double> &wholeClasses) {
    double bestAccuracy = -1;
    int iter = 0;

    double optimalC = -1;
    double optimalc = -1;

    int steps_amount = 10;
    double minc = 0, maxc = 100;
    for (double C_ : {0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0}) {
        for (int step = 0; step < steps_amount; step++) {
            cout << iter++ << '\n';
            double c = (double) step / steps_amount * (maxc - minc) + minc;

            double meanAccuracy = 0;

            for (int i = 0; i < 10; i++) {
                auto dataTest = splitOnDataAndTest(wholeDataSet, wholeClasses, 10, i);

                d = dataTest.first.first;
                target = dataTest.second.first;

                initGlobals(C_, makeLinearKernel(c));
                doShit();
                double accuracy = calcAccuracy(dataTest.first.second, dataTest.second.second);

                meanAccuracy += accuracy;

//                cout << accuracy << '\n';
            }

            meanAccuracy /= 10.0;
            cout << meanAccuracy << '\n';

            if (bestAccuracy < meanAccuracy) {
                bestAccuracy = meanAccuracy;
                optimalC = C;
                optimalc = c;
            }
        }
    }

    cout << "Linear kernel: accuracy: " << bestAccuracy << " C = " << optimalC << " c = "
         << optimalc
         << '\n';
}

void learnPolynomial(vector<vector<double>> &wholeDataSet, vector<double> &wholeClasses) {
//        Polynomial kernel: accuracy: 0.90990990990990994192 C = 0.050000000000000002776 c = 0 d = 2 alpha = 0.200000000000000011
    double bestAccuracy = -1;
    int iter = 0;

    double alpha = 0.5;
    double c = 0;

    double optimalAlpha = alpha;
    double optimalD = -1;
    double optimalc = -1;
    double optimalC = -1;

    int steps_amount = 10;
    double minc = 0, maxc = 2;
    double minalpha = -1, maxalpha = 1;
    for (double C_ : {0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0}) {
        for (int step = 0; step < steps_amount; step++) {
            for (int stepa = 0; stepa < steps_amount; stepa++) {
                double alpha = (double) stepa / steps_amount * (maxalpha - minalpha) + minalpha;
                double c = (double) step / steps_amount * (maxc - minc) + minc;


                for (int d_ = 2; d_ < 6; d_++) {
                    cout << iter++ << '\n';
//                    double d_ = (double) step_d / steps_amount * (maxd - mind) + mind;

                    double meanAccuracy = 0;

                    for (int i = 0; i < 10; i++) {
                        auto dataTest = splitOnDataAndTest(wholeDataSet, wholeClasses, 10, i);

                        d = dataTest.first.first;
                        target = dataTest.second.first;

                        initGlobals(C_, makePolynomialKernel(alpha, c, d_));
                        doShit();
                        double accuracy = calcAccuracy(d, target);
                        meanAccuracy += accuracy;

//                cout << accuracy << '\n';
                    }

                    meanAccuracy /= 10.0;
                    cout << meanAccuracy << '\n';

                    if (bestAccuracy < meanAccuracy) {
                        bestAccuracy = meanAccuracy;
                        optimalc = c;
                        optimalC = C;
                        optimalAlpha = alpha;
                        optimalD = d_;
                    }

                    cout << "alpha = " << alpha << " c = " << c << " d = " << d_ << " accuracy: " << meanAccuracy
                         << '\n';
                }
            }
        }
    }

    cout << "Polynomial kernel: accuracy: " << bestAccuracy << " C = " << optimalC
         << " c = " << optimalc
         << " d = " << optimalD
         << " alpha = " << optimalAlpha
         << '\n'; //0 0.5
}

void learnExponential(vector<vector<double>> &wholeDataSet, vector<double> &wholeClasses) {
    double bestAccuracy = -1;
    int iter = 0;

    double c = 0;

    double optimalBeta = -1;
    double optimalC = -1;

    int steps_amount = 100;
    double minc = 0, maxc = 10;
    double minbeta = 1, maxbeta = 5;
    for (double C_ : {0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0}) {
        for (int step_beta = 0; step_beta < steps_amount; step_beta++) {
            double beta = (double) step_beta / steps_amount * (maxbeta - minbeta) + minbeta;
            cout << iter++ << '\n';
//                    double d_ = (double) step_d / steps_amount * (maxd - mind) + mind;

            double meanAccuracy = 0;

            for (int i = 0; i < 10; i++) {
                auto dataTest = splitOnDataAndTest(wholeDataSet, wholeClasses, 10, i);

                d = dataTest.first.first;
                target = dataTest.second.first;

                initGlobals(C_, makeExponentialKernel(beta));
                doShit();
                double accuracy = calcAccuracy(d, target);

                meanAccuracy += accuracy;
            }

            meanAccuracy /= 10.0;
            cout << meanAccuracy << '\n';
            cout << "alpha = " << beta << "C = " << C_ << " accuracy: " << meanAccuracy << '\n';

            if (bestAccuracy < meanAccuracy) {
                bestAccuracy = meanAccuracy;
                optimalC = C;
                optimalBeta = beta;
            }
        }
    }

    cout << "Polynomial kernel: accuracy: " << bestAccuracy << " C = " << optimalC
         << " beta = " << optimalBeta
         << '\n';
}

pair<pair<vector<vector<double>>, vector<vector<double>>>, pair<vector<double>, vector<double>>>
splitOnDataAndTest(vector<vector<double>> &wholeDataset, vector<double> classes, int testsAmount, int testIndex) {
    int testSize = wholeDataset.size() / testsAmount;
    int startIndex = testSize * testIndex;

    vector<vector<double>> dataset;
    vector<vector<double>> test;

    vector<double> dataset_classes;
    vector<double> test_classes;

    for (int i = 0; i < wholeDataset.size(); i++) {
        if (i >= startIndex && i <= startIndex + testSize) {
            test.push_back(wholeDataset[i]);
            test_classes.push_back(classes[i]);
        } else {
            dataset.push_back(wholeDataset[i]);
            dataset_classes.push_back(classes[i]);
        }
    }

    return {{dataset,         test},
            {dataset_classes, test_classes}};
}

int main() {
//    ios_base::sync_with_stdio(false);
//    cin.tie(nullptr);
    cout << setprecision(20);

    freopen("chips.txt", "r", stdin);

    int n;
    cin >> n;


//    cin >> C;
//
//    double c;
//    cin >> c;

    d = vector<vector<double>>(n, vector<double>(2));
    target = vector<double>(n);

    double bestAccuracy = -1;
    pair<double, double> optimal_cC = {-1, -1};

    cinParameters();

//    cin >> optimal_cC.first >> optimal_cC.second;

    vector<vector<double>> wholeDataSet(n, vector<double>(2));
    vector<double> wholeClasses(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 2; j++) {
//            cin >> d[i][j];
            cin >> wholeDataSet[i][j];
        }
        cin >> wholeClasses[i];
//        cin >> target[i];
    }

    if (precalc) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                precalced[i][j] = kernel(d[i], d[j]);
            }
        }
    }

//    learnLinear(wholeDataSet, wholeClasses);
//    learnPolynomial(wholeDataSet, wholeClasses);
    learnExponential(wholeDataSet, wholeClasses);

//    doShit();

//    for (int i = 0; i < n; i++) {
//        cout << "was: " << target[i] << " classified: " << (classificator(d[i]) ? 1 : -1) << '\n';
//    }
//
//    cout << calcAccuracy(d, target) << '\n';
//
//    for (int i = 0; i < d.size(); i++) {
//        cout << a[i] << " ";
//    }
//    cout << b;
}