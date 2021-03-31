#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <deque>
#include <algorithm>

using namespace std;

long double pxcNgram(deque<int> const &x, int c, map<deque<int>, map<int, int>> &word2classes, map<int, int> &class2cnt,
                     long double alpha) {
    long double countX = word2classes[x][c];
    long double countC = class2cnt[c];

    return (countX + alpha) / (countC + 2.0 * alpha);
}

long double pxc(int x, int c, map<int, map<int, int>> &word2classes, map<int, int> &class2cnt,
                long double alpha) {
    long double countX = word2classes[x][c];
    long double countC = class2cnt[c];

    return (countX + alpha) / (countC + 2.0 * alpha);
}

long double py(int y, map<int, int> &class2cnt, int classesCnt) {
    return (long double) class2cnt[y] / classesCnt;
}

vector<vector<long double>> doShitNgrams(vector<vector<int>> &messages, vector<int> &target,
                                         vector<vector<int>> &messages_test,
                                         vector<long double> &lambda, long double alpha,
                                         int ngram_size) {
    int k = lambda.size();
    int n = messages.size();

    map<deque<int>, map<int, int>> word2classes;
    map<int, int> class2cnt;
    set<int> classes;

    set<deque<int>> dataset_words;

    for (int i = 0; i < n; i++) {
        classes.insert(target[i]);
        class2cnt[target[i]]++;

        set<deque<int>> words;
        deque<int> ngram;
        for (int &word : messages[i]) {
            ngram.push_back(word);
            if (ngram.size() > ngram_size) {
                ngram.pop_front();
            }

            if (ngram.size() == ngram_size) {
//                vector<int> asVec(ngram.begin(), ngram.end());
                words.insert(ngram);
                dataset_words.insert(ngram);
            }
        }

        for (auto &word : words) {
            word2classes[word][target[i]]++;
        }
    }

    int m = messages_test.size();

    vector<set<deque<int>>> words(m);

    for (int i = 0; i < m; i++) {
        deque<int> ngram;
        for (int j = 0; j < messages_test[i].size(); j++) {
            ngram.push_back(messages_test[i][j]);
            if (ngram.size() > ngram_size) {
                ngram.pop_front();
            }

            if (ngram.size() == ngram_size) {
                words[i].insert(ngram);
            }
        }
    }

    map<pair<deque<int>, int>, double> probCache;

    for (int c : classes) {
        for (deque<int> const &word : dataset_words) {
            long double wordProb = pxcNgram(word, c, word2classes, class2cnt, alpha);
            probCache[{word, c}] = wordProb;
        }
    }


    int test = dataset_words.size();
    map<int, map<int, long double>> pcipXci;

    for (int i = 0; i < m; i++) {
//        cout << i << '\n';
//        cout.flush();
        for (int c : classes) {
            long double p = lambda[c] * py(c, class2cnt, n);

            for (deque<int> const &word : dataset_words) {
                long double wordProb = probCache[{word, c}];// pxc(word, c, word2classes, class2cnt, alpha);

                set<deque<int>> &wordsI = words[i];

                if (wordsI.find(word) == wordsI.end()) {
                    p *= (1.0 - wordProb);
                } else {
                    p *= wordProb;
                }
            }

            pcipXci[i][c] = p;
        }
    }

    vector<vector<long double>> res(m, vector<long double>(k));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int c = j;

            if (classes.find(c) == classes.end()) {
                res[i][j] = 0;
//                cout << 0 << " ";
                continue;
            }

            long double p = pcipXci[i][c];
            long double div = 0.0;

            for (int c1 : classes) {
                div += pcipXci[i][c1];
            }

            p /= div;

            res[i][j] = p;
//            cout << p << " ";
        }
//        cout << '\n';
    }

    return res;
}

vector<vector<long double>> doShit(vector<vector<int>> &messages, vector<int> &target,
                                   vector<vector<int>> &messages_test,
                                   vector<long double> &lambda, long double alpha,
                                   int mx) {
    int k = lambda.size();
    int n = messages.size();

    map<int, map<int, int>> word2classes;
    map<int, int> class2cnt;
    set<int> classes;

    set<int> dataset_words;

    for (int i = 0; i < n; i++) {
        classes.insert(target[i]);
        class2cnt[target[i]]++;

        set<int> words;
        for (int &word : messages[i]) {
            words.insert(word);
            dataset_words.insert(word);
        }

        for (auto &word : words) {
            word2classes[word][target[i]]++;
        }
    }

    int m = messages_test.size();

    vector<set<int>> words(m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < messages_test[i].size(); j++) {
            words[i].insert(messages_test[i][j]);
        }
    }

    vector<vector<double>> probCache(mx, vector<double>(classes.size()));
    for (int c : classes) {
        for (int word : dataset_words) {
            long double wordProb = pxc(word, c, word2classes, class2cnt, alpha);
            probCache[word][c] = wordProb;
        }
    }


    int test = dataset_words.size();
    map<int, map<int, long double>> pcipXci;

    for (int i = 0; i < m; i++) {
//        cout << i << '\n';
//        cout.flush();
        for (int c : classes) {
            long double p = lambda[c] * py(c, class2cnt, n);

            for (int word : dataset_words) {
                long double wordProb = probCache[word][c];//pxc(word, c, word2classes, class2cnt, alpha);
                set<int> &wordsI = words[i];

                if (wordsI.find(word) == wordsI.end()) {
                    p *= (1.0 - wordProb);
                } else {
                    p *= wordProb;
                }
            }

            pcipXci[i][c] = p;
        }
    }

    vector<vector<long double>> res(m, vector<long double>(k));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int c = j;

            if (classes.find(c) == classes.end()) {
                res[i][j] = 0;
//                cout << 0 << " ";
                continue;
            }

            long double p = pcipXci[i][c];
            long double div = 0.0;

            for (int c1 : classes) {
                div += pcipXci[i][c1];
            }

            p /= div;

            res[i][j] = p;
//            cout << p << " ";
        }
//        cout << '\n';
    }

    return res;
}

double compareWithTestTargets(vector<vector<long double>> &probabilities, vector<int> &test_target) {
    assert(probabilities.size() == test_target.size());

    double hits = 0;
    double misses = 0;

    for (int i = 0; i < probabilities.size(); i++) {
        pair<long double, int> predicted_class = {-1, -1};

        for (int j = 0; j < probabilities[i].size(); j++) {
            predicted_class = max(predicted_class, {probabilities[i][j], j});
        }

        if (predicted_class.second == test_target[i]) {
            hits++;
        } else {
            misses++;
        }

//        cout << "actual: " << test_target[i] << " predicted: " << predicted_class.second << " with prob: "
//             << predicted_class.first << '\n';
    }
//    cout << "accuracy: " << hits / (hits + misses);

    return hits / (hits + misses);
}

bool isAllNonSpamClassifiedCorrectly(vector<vector<long double>> &probabilities, vector<int> &test_target) {
    assert(probabilities.size() == test_target.size());

    for (int i = 0; i < probabilities.size(); i++) {
        pair<long double, int> predicted_class = {-1, -1};

        for (int j = 0; j < probabilities[i].size(); j++) {
            predicted_class = max(predicted_class, {probabilities[i][j], j});
        }

        if (predicted_class.second == 0 && test_target[i] == 1) {
            return false;
        }
    }

    return true;
}

pair<double, double> calcROCRow(vector<vector<long double>> &probabilities, vector<int> &test_target) {
    assert(probabilities.size() == test_target.size());

    double tn = 0;
    double tp = 0;
    double fp = 0;
    double fn = 0;

    for (int i = 0; i < probabilities.size(); i++) {
        pair<long double, int> predicted_class = {-1, -1};

        for (int j = 0; j < probabilities[i].size(); j++) {
            predicted_class = max(predicted_class, {probabilities[i][j], j});
        }

        if (predicted_class.second == 1 && test_target[i] == 1) {
            tp++;
        }
        if (predicted_class.second == 0 && test_target[i] == 0) {
            tn++;
        }
        if (predicted_class.second == 1 && test_target[i] == 0) {
            fp++;
        }
        if (predicted_class.second == 0 && test_target[i] == 1) {
            fn++;
        }
    }

    return {fp / (fp + tn), tp / (tp + fn)};
}

vector<int> mergeSubjectIntoText(vector<int> &subject, vector<int> &text) {
    vector<int> res(subject.size() + text.size());

    for (int i = 0; i < subject.size(); i++) {
        res[i] = -subject[i];
    }
    for (int i = 0; i < text.size(); i++) {
        res[subject.size() + i] = text[i];
    }

    return res;
}

vector<vector<vector<int>>> mergeSubjectsIntoTexts(
        vector<vector<vector<int>>> &part2text,
        vector<vector<vector<int>>> &part2subject) {
    assert(part2text.size() == part2subject.size());

    vector<vector<vector<int>>> res(part2text.size());

    for (int i = 0; i < part2text.size(); i++) {
        res[i].resize(part2text[i].size());
        for (int j = 0; j < part2text[i].size(); j++) {
            res[i][j] = mergeSubjectIntoText(part2subject[i][j], part2text[i][j]);
        }
    }

    return res;
}

pair<vector<vector<int>>, vector<int>> mergePartsExcludingOne(vector<vector<vector<int>>> &part2words,
                                                              vector<vector<int>> &part2target,
                                                              int exclude) {
    vector<vector<int>> words;
    vector<int> target;

    for (int i = 0; i < part2words.size(); i++) {
        if (i == exclude) {
            continue;
        }

        for (int j = 0; j < part2words[i].size(); j++) {
            words.push_back(part2words[i][j]);
            target.push_back(part2target[i][j]);
        }
    }

    return {words, target};
}

pair<map<deque<int>, int>, int> extractFeatures(int ngramSize, vector<vector<int>> &words) {
    map<deque<int>, int> res;
    int mx = 0;

    for (int i = 0; i < words.size(); i++) {
        deque<int> curNGram;

        if (words[i].size() < ngramSize) {
            curNGram = {words[i].begin(), words[i].end()};
            res[curNGram] = mx++;
            continue;
        }

        for (int j = 0; j < words[i].size(); j++) {
            curNGram.push_back(words[i][j]);
            if (curNGram.size() > ngramSize) {
                curNGram.pop_front();
            }

            if (curNGram.size() == ngramSize) {
                res[curNGram] = mx++;
            }
        }
    }

    return {res, mx};
}

vector<vector<int>> convert2ngrams(vector<vector<int>> &words, int ngramSize, map<deque<int>, int> &ngrams) {
    vector<vector<int>> res(words.size());

    for (int i = 0; i < words.size(); i++) {
        deque<int> curNGram;

        for (int j = 0; j < words[i].size(); j++) {
            curNGram.push_back(words[i][j]);
            if (curNGram.size() > ngramSize) {
                curNGram.pop_front();
            }

            if (curNGram.size() == ngramSize) {
                if (ngrams.find(curNGram) != ngrams.end()) {
                    int ngramId = ngrams[curNGram];
                    res[i].push_back(ngramId);
                }
            }
        }
    }

    return res;
}

void printROC(int ngramSize, vector<long double> &lambdas,
              vector<vector<vector<int>>> &partsSubjectTextMerged,
              vector<vector<int>> &part2targets) {
    auto wordsTarget = mergePartsExcludingOne(partsSubjectTextMerged, part2targets, 0);
    auto featuresMx = extractFeatures(ngramSize, wordsTarget.first);
    auto wordsConverted = convert2ngrams(wordsTarget.first, ngramSize, featuresMx.first);
    auto testConverted = convert2ngrams(partsSubjectTextMerged[0], ngramSize, featuresMx.first);

    vector<pair<double, double>> fpr2tpr;

//        double min_alpha = 0, max_alpha = 1;
    for (double alpha : {1e-20, 1e-15,
                         1e-10, 5e-9, 1e-9, 5e-8, 1e-8, 5e-7, 1e-7, 5e-6, 1e-6, 6e-5, 1e-5, 5e-3, 1e-4, 5e-3, 1e-3,
                         5e-2, 1e-2, 5e-1, 1e-1,
                         0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                         11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                         100.0}) {
//        for (int step = 0; step <= 100; step++) {
//            double alpha = ((double) step / 100) * (max_alpha - min_alpha) + min_alpha;
//            int excluded = 0;

//            auto &partsMerged = partsMergedE[excluded];// mergePartsExcludingOne(partsSubjectTextMerged, part2targets, excluded);
        auto &target = wordsTarget.second;
        auto &words = wordsConverted;

//            map<deque<int>, int> featuresExtracted = extractFeatures(ngramSize, words);
//            vector<vector<int>> &wordsConverted = wordsConvertedE[excluded];// convert2ngrams(words, ngramSize, featuresExtracted);
//            vector<vector<int>> &testConverted = testConvertedE[excluded];// convert2ngrams(partsSubjectTextMerged[excluded], ngramSize, featuresExtracted);

//            int test = featuresExtracted.size(), wordsSize = wordsConverted.size(), testSize = testConverted.size();
        vector<vector<long double>> shit = doShit(wordsConverted, target, testConverted,
                                                  lambdas, alpha, featuresMx.second);

        auto FPR_TPR = calcROCRow(shit, part2targets[0]);
        fpr2tpr.push_back(FPR_TPR);
    }

    sort(fpr2tpr.begin(), fpr2tpr.end());

    for (auto &p : fpr2tpr) {
        cout << p.first << "," << p.second << '\n';
        cout.flush();
    }
}

void learn(int ngramSize, vector<long double> &lambdas,
           vector<vector<vector<int>>> &partsSubjectTextMerged, vector<vector<int>> &part2targets) {
    pair<double, double> bestAccuracy = {-1, -1};

    vector<pair<vector<vector<int>>, vector<int>>> partsMergedE(10);
    vector<vector<vector<int>>> wordsConvertedE(10);
    vector<vector<vector<int>>> testConvertedE(10);
    vector<int> featureMx(10);
    for (int excluded = 0; excluded < 10; excluded++) {
        partsMergedE[excluded] = mergePartsExcludingOne(partsSubjectTextMerged, part2targets, excluded);
        auto &target = partsMergedE[excluded].second;
        auto &words = partsMergedE[excluded].first;

        auto featuresMx = extractFeatures(ngramSize, words);
        map<deque<int>, int> featuresExtracted = featuresMx.first;
        featureMx[excluded] = featuresMx.second;
        wordsConvertedE[excluded] = convert2ngrams(words, ngramSize, featuresExtracted);
        testConvertedE[excluded] = convert2ngrams(partsSubjectTextMerged[excluded], ngramSize,
                                                  featuresExtracted);
    }

// (ngramSize=1) Best accuracy: 0.97155963302752312849 with alpha: 0.0006 (0.00059999999999999994744)
// (ngramSize=2) Best accuracy: 0.96788990825688081543 with alpha: 1.0000000000000000209e-08
// (ngramSize=3) Best accuracy: 0.9458715596330276032 with alpha: 1e-11 (9.999999999999999395e-12)
    double min_alpha = 0, max_alpha = 1e-7;
    for (int step = 1; step < 11; step++) {
        cout << step << '\n';
        cout.flush();

        double alpha = ((double) step / 10) * (max_alpha - min_alpha) + min_alpha;

        cout << "trying alpha: " << alpha << '\n';

        double meanAccuracy = 0;
        for (int excluded = 0; excluded < 10; excluded++) {
            auto &partsMerged = partsMergedE[excluded];// mergePartsExcludingOne(partsSubjectTextMerged, part2targets, excluded);
            auto &target = partsMerged.second;
            auto &words = partsMerged.first;

//            map<deque<int>, int> featuresExtracted = extractFeatures(ngramSize, words);
            vector<vector<int>> &wordsConverted = wordsConvertedE[excluded];// convert2ngrams(words, ngramSize, featuresExtracted);
            vector<vector<int>> &testConverted = testConvertedE[excluded];// convert2ngrams(partsSubjectTextMerged[excluded], ngramSize, featuresExtracted);

//            int test = featuresExtracted.size(), wordsSize = wordsConverted.size(), testSize = testConverted.size();
            vector<vector<long double>> shit = doShit(wordsConverted, target, testConverted,
                                                      lambdas, alpha, featureMx[excluded]);

            double accuracy = compareWithTestTargets(shit, part2targets[excluded]);
            meanAccuracy += accuracy;
        }

        meanAccuracy /= 10.0;
        bestAccuracy = max(bestAccuracy, {meanAccuracy, alpha});
    }

    cout << "Best accuracy: " << bestAccuracy.first << " with alpha: " << bestAccuracy.second;
}

void findLambdas(int ngramSize, double alpha,
                 vector<vector<vector<int>>> &partsSubjectTextMerged, vector<vector<int>> &part2targets) {
    pair<double, double> bestAccuracy = {-1, -1};

    vector<long double> lambdas = {1, 1};

    vector<pair<vector<vector<int>>, vector<int>>> partsMergedE(10);
    vector<vector<vector<int>>> wordsConvertedE(10);
    vector<vector<vector<int>>> testConvertedE(10);
    vector<int> featureMx(10);
    for (int excluded = 0; excluded < 10; excluded++) {
        partsMergedE[excluded] = mergePartsExcludingOne(partsSubjectTextMerged, part2targets, excluded);
        auto &target = partsMergedE[excluded].second;
        auto &words = partsMergedE[excluded].first;

        auto featuresMx = extractFeatures(ngramSize, words);
        map<deque<int>, int> featuresExtracted = featuresMx.first;
        featureMx[excluded] = featuresMx.second;
        wordsConvertedE[excluded] = convert2ngrams(words, ngramSize, featuresExtracted);
        testConvertedE[excluded] = convert2ngrams(partsSubjectTextMerged[excluded], ngramSize,
                                                  featuresExtracted);
    }

// best is 1e60
    long double min_lambda = 1, max_lambda = 1e45;
    for (int step = 0; step <= 100; step++) {
//    int step = 10;
//        cout << step << '\n';
//        cout.flush();

        long double lambda = ((long double) step / 100.0L) * (max_lambda - min_lambda) + min_lambda;
        lambdas[1] = lambda;

//        cout << "trying lambda[1] = " << lambda << '\n';

        double meanAccuracy = 0;
        bool isOk = true;
        for (int excluded = 0; excluded < 10; excluded++) {
            auto &partsMerged = partsMergedE[excluded];
            auto &target = partsMerged.second;
            auto &words = partsMerged.first;

            vector<vector<int>> &wordsConverted = wordsConvertedE[excluded];
            vector<vector<int>> &testConverted = testConvertedE[excluded];

            vector<vector<long double>> shit = doShit(wordsConverted, target, testConverted,
                                                      lambdas, alpha, featureMx[excluded]);

            double accuracy = compareWithTestTargets(shit, part2targets[excluded]);
            isOk = isAllNonSpamClassifiedCorrectly(shit, part2targets[excluded]);

//            if (!isOk) {
//                isOk = isAllNonSpamClassifiedCorrectly(shit, part2targets[excluded]);
//                break;
//            }

            meanAccuracy += accuracy;
        }

        meanAccuracy /= 10.0;
        cout << lambda << "," << meanAccuracy << "," << isOk << '\n';
        cout.flush();

        if (isOk) {
            bestAccuracy = max(bestAccuracy, {meanAccuracy, lambda});
        }
    }

//    cout << "Best accuracy: " << bestAccuracy.first << " with alpha: " << bestAccuracy.second;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

//    int test;
//    cin >> test;

    freopen("dataset.txt", "r", stdin);

    int parts_amount;

    cin >> parts_amount;

    vector<vector<vector<int>>> part2subjects(parts_amount);
    vector<vector<vector<int>>> part2texts(parts_amount);
    vector<vector<int>> part2targets(parts_amount);

    for (int i = 0; i < parts_amount; i++) {
        int part_size;
        cin >> part_size;

        part2subjects[i].resize(part_size);
        part2texts[i].resize(part_size);
        part2targets[i].resize(part_size);

        for (int j = 0; j < part_size; j++) {
            int subject_size;
            cin >> subject_size;

            vector<int> subject(subject_size);
            for (int k = 0; k < subject_size; k++) {
                cin >> subject[k];
            }

            int text_size;
            cin >> text_size;

            vector<int> text(text_size);
            for (int k = 0; k < text_size; k++) {
                cin >> text[k];
            }

            int target;
            cin >> target;

            part2targets[i][j] = target;
            part2texts[i][j] = text;
            part2subjects[i][j] = subject;
        }
    }

    vector<long double> lambdas = {1, 1};

    int ngramSize = 1;

    vector<vector<vector<int>>> partsSubjectTextMerged = mergeSubjectsIntoTexts(part2texts, part2subjects);

//    printROC(ngramSize, lambdas, partsSubjectTextMerged, part2targets);
//    learn(ngramSize, lambdas, partsSubjectTextMerged, part2targets);
    findLambdas(ngramSize, 0.0006, partsSubjectTextMerged, part2targets);
}