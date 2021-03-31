#include <iostream>
#include <cassert>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>

using namespace std;

vector<vector<long double>> matmul(vector<vector<long double>> &a, vector<vector<long double>> &b) {
    assert(a[0].size() == b.size());

    vector<vector<long double>> res(a.size(), vector<long double>(b[0].size()));

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b[0].size(); j++) {
            for (int k = 0; k < b.size(); k++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return res;
}

vector<vector<long double>> matadd(vector<vector<long double>> &a, vector<vector<long double>> &b) {
    assert(a.size() == b.size());
    assert(a[0].size() == b[0].size());

    vector<vector<long double>> res(a.size(), vector<long double>(a[0].size()));

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            res[i][j] = a[i][j] + b[i][j];
        }
    }

    return res;
}

vector<vector<long double>> adamar(vector<vector<long double>> &a, vector<vector<long double>> &b) {
    assert(a.size() == b.size());
    assert(a[0].size() == b[0].size());

    vector<vector<long double>> res(a.size(), vector<long double>(a[0].size()));

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            res[i][j] = a[i][j] * b[i][j];
        }
    }

    return res;
}

vector<vector<long double>> transpose(vector<vector<long double>> &m) {
    vector<vector<long double>> res(m[0].size(), vector<long double>(m.size()));

    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            res[j][i] = m[i][j];
        }
    }

    return res;
}

void print_mat(const vector<vector<long double>> &mat) {
    for (auto &row :mat) {
        for (long double v : row) {
            cout << v << " ";
        }
        cout << '\n';
    }
}

class neuron;

map<neuron *, vector<vector<long double>>> end2der;

class neuron {
public:
    vector<neuron *> ancestors;
    vector<neuron *> children;

    neuron() {}

    neuron(const vector<neuron *> &ancestors, const vector<neuron *> &children) : ancestors(ancestors),
                                                                                  children(children) {
    }

    void addChild(neuron *ch) {
        children.push_back(ch);
    }

    vector<vector<long double>> calc_der(neuron *who) {
        vector<vector<long double>> childrenDerivative;

        if (end2der.find(this) != end2der.end()) {
            childrenDerivative = end2der[this];
        }

        if (!children.empty()) {
            auto ch0 = children[0]->calc_der(this);
            if (childrenDerivative.empty()) {
                childrenDerivative = ch0;
            } else {
                childrenDerivative = matadd(childrenDerivative, ch0);
            }

            for (int i = 1; i < children.size(); i++) {
                auto nthChild = children[i]->calc_der(this);

                childrenDerivative = matadd(childrenDerivative, nthChild);
            }
        }

        return calc_derivative(who, childrenDerivative);
    }

    virtual vector<vector<long double>> calc_value() = 0;

    vector<vector<long double>> value_memo;

    vector<vector<long double>> calc_value_memo() {
        if (value_memo.empty()) {
            value_memo = calc_value();
        }

        return value_memo;
    }

    virtual vector<vector<long double>> calc_derivative(neuron *who, vector<vector<long double>> der) = 0;

    virtual pair<int, int> calc_size() = 0;
};

class input : public neuron {
public:
    int r = -1;
    int c = -1;
    vector<vector<long double>> value;

    input(int r, int c) : r(r),
                          c(c) {}

    void setValue(vector<vector<long double>> &value) {
        assert(r != -1);
        assert(c != -1);
        assert(value.size() == r);
        assert(value[0].size() == c);

        this->value = value;
    }

    vector<vector<long double>> calc_value() override {
        return value;
    }

    vector<vector<long double>> calc_derivative(neuron *who, vector<vector<long double>> der) override {
        return der;
    }

    pair<int, int> calc_size() override {
        return {value.size(), value[0].size()};
    }
};

class tnh : public neuron {
public:
    explicit tnh(neuron *parent) : neuron({parent}, {}) {}

    vector<vector<long double>> calc_value() override {
        auto parentValue = ancestors[0]->calc_value_memo();

        for (auto &row : parentValue) {
            for (long double &cell : row) {
                cell = tanh(cell);
            }
        }

        return parentValue;
    }

    vector<vector<long double>> calc_derivative(neuron *who, vector<vector<long double>> childDerivative) override {
        auto value = calc_value_memo();

        assert(childDerivative.size() == value.size());
        assert(childDerivative[0].size() == value[0].size());

        for (int i = 0; i < value.size(); i++) {
            for (int j = 0; j < value[0].size(); j++) {
                childDerivative[i][j] *= (1 - value[i][j] * value[i][j]);
            }
        }

        return childDerivative;
    }

    pair<int, int> calc_size() override {
        return ancestors[0]->calc_size();
    }
};

class rlu : public neuron {
public:
    long double alpha;

    explicit rlu(long double alpha, neuron *parent) : neuron({parent}, {}), alpha(alpha) {}

    vector<vector<long double>> calc_value() override {
        auto parentValue = ancestors[0]->calc_value_memo();

        for (auto &row : parentValue) {
            for (long double &cell : row) {
                cell = cell >= 0 ? cell : alpha * cell;
            }
        }

        return parentValue;
    }

    vector<vector<long double>> calc_derivative(neuron *who, vector<vector<long double>> childDerivative) override {
        auto value = calc_value_memo();

        assert(childDerivative.size() == value.size());
        assert(childDerivative[0].size() == value[0].size());

        for (int i = 0; i < value.size(); i++) {
            for (int j = 0; j < value[0].size(); j++) {
                childDerivative[i][j] *= value[i][j] >= 0 ? 1 : alpha;
            }
        }

        return childDerivative;
    }

    pair<int, int> calc_size() override {
        return ancestors[0]->calc_size();
    }
};


class mul : public neuron {
public:
    vector<vector<long double>> a;
    vector<vector<long double>> b;

    mul(neuron *a, neuron *b) : neuron({a, b}, {}) {}

    vector<vector<long double>> calc_value() override {
        a = ancestors[0]->calc_value_memo();
        b = ancestors[1]->calc_value_memo();

        return matmul(a, b);
    }

    vector<vector<long double>> calc_derivative(neuron *who, vector<vector<long double>> d) override {
        assert(who == ancestors[0] || who == ancestors[1]);

        if (a.empty()) {
            a = ancestors[0]->calc_value_memo();
            b = ancestors[1]->calc_value_memo();
        }

        if (who == ancestors[0]) {
            auto b_tr = transpose(b);
            return matmul(d, b_tr);
        }

        auto a_tr = transpose(a);
        return matmul(a_tr, d);
    }

    pair<int, int> calc_size() override {
        auto asz = ancestors[0]->calc_size();
        auto bsz = ancestors[1]->calc_size();

        return {asz.first, bsz.second};
    }
};

class sum : public neuron {
public:
    sum(vector<neuron *> &a) : neuron(a, {}) {}

    vector<vector<long double>> calc_value() override {
        vector<vector<long double>> res = ancestors[0]->calc_value_memo();

        for (int i = 1; i < ancestors.size(); i++) {
            auto nthanc = ancestors[i]->calc_value_memo();
            res = matadd(res, nthanc);
        }

        return res;
    }

    vector<vector<long double>> calc_derivative(neuron *who, vector<vector<long double>> der) override {
        return der;
    }

    pair<int, int> calc_size() override {
        return ancestors[0]->calc_size();
    }
};

class had : public neuron {
public:
    had(vector<neuron *> &a) : neuron(a, {}) {}

    vector<vector<long double>> calc_value() override {
        vector<vector<long double>> res = ancestors[0]->calc_value_memo();

        for (int i = 1; i < ancestors.size(); i++) {
            auto nthanc = ancestors[i]->calc_value_memo();
            res = adamar(res, nthanc);
        }

        return res;
    }

    vector<vector<long double>> calc_derivative(neuron *who, vector<vector<long double>> der) override {
        vector<vector<long double>> res = ancestors[0]->calc_value_memo();

        int chIndex = -1;
        for (int i = 0; i < ancestors.size(); i++) {
            if (ancestors[i] == who) {
                chIndex = i;
            }
        }

        assert(chIndex != -1);

        res = vector<vector<long double>>(res.size(), vector<long double>(res[0].size(), 1.0));

        for (int i = 0; i < ancestors.size(); i++) {
            if (i == chIndex) {
                continue;
            }

            auto nthanc = ancestors[i]->calc_value_memo();
            res = adamar(res, nthanc);
        }

        return adamar(res, der);
    }

    pair<int, int> calc_size() override {
        return ancestors[0]->calc_size();
    }
};


//class output : public hidden {
//public:
//    long double value;
//
//    output(const vector<neuron *> &ancestors) : hidden(ancestors) {}
//
//    long double calc_derivative() override {
//        vector<>
//        for (int i = 0; i < ancestors.size(); i++) {
//
//        }
//
//        return 1.0;
//    }
//};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);
//    freopen("L.txt", "r", stdin);

    int n, m, k;
    cin >> n >> m >> k;

    vector<input *> inputs;
    vector<neuron *> i2n(n + 1);

    for (int i = 0; i < n; i++) {
        string type;
        cin >> type;

        neuron *new_neuron = nullptr;

        if (type == "var") {
            int r, c;
            cin >> r >> c;

            new_neuron = new input(r, c);
        }

        if (type == "tnh") {
            int neuron_index;
            cin >> neuron_index;

            new_neuron = new tnh(i2n[neuron_index]);

            i2n[neuron_index]->addChild(new_neuron);
        }

        if (type == "rlu") {
            long double alpha;
            int neuron_index;
            cin >> alpha >> neuron_index;

            new_neuron = new rlu(1.0 / alpha, i2n[neuron_index]);

            i2n[neuron_index]->addChild(new_neuron);
        }

        if (type == "mul") {
            int a_index, b_index;
            cin >> a_index >> b_index;

            new_neuron = new mul(i2n[a_index], i2n[b_index]);

            i2n[a_index]->addChild(new_neuron);
            i2n[b_index]->addChild(new_neuron);
        }

        if (type == "sum") {
            int len;
            cin >> len;

            vector<neuron *> sons(len);
            for (int j = 0; j < len; j++) {
                int index;
                cin >> index;

                sons[j] = i2n[index];
            }

            new_neuron = new sum(sons);

            for (neuron *son : sons) {
                son->addChild(new_neuron);
            }
        }

        if (type == "had") {
            int len;
            cin >> len;

            vector<neuron *> sons(len);
            for (int j = 0; j < len; j++) {
                int index;
                cin >> index;

                sons[j] = i2n[index];
            }

            new_neuron = new had(sons);

            for (neuron *son : sons) {
                son->addChild(new_neuron);
            }
        }

        assert(new_neuron != nullptr);

        i2n[i + 1] = new_neuron; // +1 or not
    }

    for (int i = 0; i < m; i++) {
        auto *curr_input = dynamic_cast<input *>(i2n[i + 1]);
        int r = curr_input->r;
        int c = curr_input->c;

        vector<vector<long double>> value(r, vector<long double>(c));

        for (int j = 0; j < r; j++) {
            for (int l = 0; l < c; l++) {
                cin >> value[j][l];
            }
        }

        curr_input->setValue(value);
    }

    for (int i = 0; i < k; i++) {
        auto sz = i2n[n + 1 - k + i]->calc_size();

        vector<vector<long double>> value(sz.first, vector<long double>(sz.second));

        for (int j = 0; j < sz.first; j++) {
            for (int l = 0; l < sz.second; l++) {
                cin >> value[j][l];
            }
        }

        end2der[i2n[n + 1 - k + i]] = value;
    }
//    for (int i = 0; i < k; i++) {
//        end2der[i2n[n - k + i]] =
//    }

    for (int i = 0; i < k; i++) {
        print_mat(i2n[n + 1 - k + i]->calc_value_memo());
    }
//    print_mat(i2n[n]->calc_value_memo());

    for (int i = 0; i < m; i++) {
        print_mat(i2n[i + 1]->calc_der(i2n[i + 1]));
    }

    int test = 1 + 1;
}