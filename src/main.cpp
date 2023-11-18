#include <limits.h>
#include <stdint.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "bitmap.h"

#define UNUSED(X) (void)(X)

#define N_TRAIN 10000
#define N_TEST 10000

#define N_0 784
#define N_1 64
#define N_2 10

Eigen::MatrixXf w_01;
Eigen::MatrixXf w_12;
Eigen::VectorXf b_1;
Eigen::VectorXf b_2;

Eigen::VectorXf costFuncPrime(Eigen::VectorXf a, Eigen::VectorXf y) {
    return a - y;
}

Eigen::VectorXf activationFunc(Eigen::VectorXf x) {
    //return 1.0f / (1.0f + (-x).array().exp());

    Eigen::VectorXf a(x.rows());
    for (int i = 0; i < x.rows(); i++) {
        if (x(i) > 0.0f) {
            a(i) = 0.25f * x(i);
        } else {
            a(i) = 0.01f * x(i);
        }
    }

    return a;
}

Eigen::VectorXf activationFuncPrime(Eigen::VectorXf x) {
    //return x.array().exp() / (x.array().exp() + 1).pow(2.0f);

    Eigen::VectorXf a(x.rows());
    for (int i = 0; i < x.rows(); i++) {
        if (x(i) > 0.0f) {
            a(i) = 0.25f;
        } else {
            a(i) = 0.01f;
        }
    }

    return a;
}

std::tuple<Eigen::VectorXf, Eigen::VectorXf> forwardPass(Eigen::VectorXf a, Eigen::MatrixXf w, Eigen::VectorXf b) {
    Eigen::VectorXf xNext = w * a + b;
    Eigen::VectorXf aNext = activationFunc(xNext);
    return std::make_tuple(xNext, aNext);
}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::VectorXf, Eigen::VectorXf> backpropagation(Eigen::VectorXf image, Eigen::VectorXf label) {
    // Go forward to last layer
    auto layer_1 = forwardPass(image, w_01, b_1);
    Eigen::VectorXf x_1 = std::get<0>(layer_1);
    Eigen::VectorXf a_1 = std::get<1>(layer_1);

    auto layer_2 = forwardPass(a_1, w_12, b_2);
    Eigen::VectorXf x_2 = std::get<0>(layer_2);
    Eigen::VectorXf a_2 = std::get<1>(layer_2);

    // dC/dx2 = dC/da_2 * da_2/dx2
    Eigen::VectorXf dc_dx2 = costFuncPrime(a_2, label).array() * activationFuncPrime(x_2).array();
    
    // dC/dx1 = dC/dx2 * dx2/da1 * da1/dx1 = dC/dx2 * w_12 * da1/dx1
    Eigen::VectorXf dc_dx1 = (w_12.transpose() * dc_dx2).array() * activationFuncPrime(x_1).array();
    
    // dC/dw = dC/dx * dx/dw = dC/dx * a
    Eigen::MatrixXf partial_w_01 = dc_dx1 * image.transpose ();
    Eigen::MatrixXf partial_w_12 = dc_dx2 * a_1.transpose ();

    // dC/db = dC/dx * dx/db = dC/dx
    Eigen::VectorXf partial_b_1 = dc_dx1;
    Eigen::VectorXf partial_b_2 = dc_dx2;

    return std::make_tuple(partial_w_01, partial_w_12, partial_b_1, partial_b_2);
}

void gradientDescent(std::vector<Eigen::VectorXf> images, std::vector<Eigen::VectorXf> labels, int N, int minibatchSize, float rate) {

    for (int i = 0; i < N; i++) {
        printf("Iteration %i!\n", i);

        for (size_t j = 0; j < images.size(); j += minibatchSize) {
            Eigen::MatrixXf partial_w_01 = Eigen::MatrixXf::Zero(N_1, N_0);
            Eigen::MatrixXf partial_w_12 = Eigen::MatrixXf::Zero(N_2, N_1);
            Eigen::VectorXf partial_b_1 = Eigen::VectorXf::Zero(N_1);
            Eigen::VectorXf partial_b_2 = Eigen::VectorXf::Zero(N_2);

            for (size_t k = j; k < j + minibatchSize; k++) {
                auto partial = backpropagation(images[k], labels[k]);
                partial_w_01 += std::get<0>(partial) / minibatchSize;
                partial_w_12 += std::get<1>(partial) / minibatchSize;
                partial_b_1 += std::get<2>(partial) / minibatchSize;
                partial_b_2 += std::get<3>(partial) / minibatchSize;
            }

            w_01 -= rate * partial_w_01;
            w_12 -= rate * partial_w_12;
            b_1 -= rate * partial_b_1;
            b_2 -= rate * partial_b_2;
        }
    }
}

float evaluate(std::vector<Eigen::VectorXf> images, std::vector<Eigen::VectorXf> labels) {
    int correct = 0;

    for (size_t i = 0; i < images.size(); i++) {
        auto layer_1 = forwardPass(images[i], w_01, b_1);
        Eigen::VectorXf x_1 = std::get<0>(layer_1);
        Eigen::VectorXf a_1 = std::get<1>(layer_1);

        auto layer_2 = forwardPass(a_1, w_12, b_2);
        Eigen::VectorXf x_2 = std::get<0>(layer_2);
        Eigen::VectorXf a_2 = std::get<1>(layer_2);

        int indexCorrect = 0;
        for (int j = 0; j < N_2; j++) {
            if (labels[i](j) == 1.0f) {
                indexCorrect = j;
            }
        }

        int indexMax = 0;
        for (int j = 0; j < N_2; j++) {
            if (a_2[j] > a_2[indexMax]) {
                indexMax = j;
            }
        }

        if (indexCorrect == indexMax) {
            correct++;
        }
    }

    return 100.0f * correct / images.size();
}

int main(int argc, char** argv) {
    UNUSED(argc);
    UNUSED(argv);

    std::vector<Eigen::VectorXf> trainImages;
    std::vector<Eigen::VectorXf> trainLabels;
    std::vector<Eigen::VectorXf> testImages;
    std::vector<Eigen::VectorXf> testLabels;

    /* Load Data */
    printf("Opening Files!\n");
    for (size_t i = 0; i < N_TRAIN; i++) {
        char filepath[PATH_MAX];
        Image img;
        Eigen::VectorXf image(N_0);
        Eigen::VectorXf label(N_2);

        // Import Image
        sprintf(filepath, ".\\res\\%05zi.bmp", i);
        importBMP(filepath, img);

        // Create Image Vector
        for (int j = 0; j < N_0; j++) {
            image(j) = img.data[j * 3] / 255.0f;
        }

        // Create Label Vector
        for (int j = 0; j < N_2; j++) {
            if (j == img.label) {
                label(j) = 1.0f;
            } else {
                label(j) = 0.0f;
            }
        }
    
        // Add to Vector
        trainImages.push_back(image);
        trainLabels.push_back(label);

        // Free memory
        delete img.data;
    }

    for (size_t i = N_TRAIN; i < N_TRAIN + N_TEST; i++) {
        char filepath[PATH_MAX];
        Image img;
        Eigen::VectorXf image(N_0);
        Eigen::VectorXf label(N_2);

        // Import Image
        sprintf(filepath, ".\\res\\%05zi.bmp", i);
        importBMP(filepath, img);

        // Create Image Vector
        for (int j = 0; j < N_0; j++) {
            image(j) = img.data[j * 3] / 255.0f;
        }

        // Create Label Vector
        for (int j = 0; j < N_2; j++) {
            if (j == img.label) {
                label(j) = 1.0f;
            } else {
                label(j) = 0.0f;
            }
        }

        // Add to Vector
        testImages.push_back(image);
        testLabels.push_back(label);

        // Free memory
        delete img.data;
    }

    /* Train Neural Network */
    printf("\nTraining Neural Network!\n");
    w_01 = Eigen::MatrixXf::Random(N_1, N_0);
    w_12 = Eigen::MatrixXf::Random(N_2, N_1);
    b_1 = Eigen::VectorXf::Zero(N_1);
    b_2 = Eigen::VectorXf::Zero(N_2);

    gradientDescent(trainImages, trainLabels, 10, 100, 1.0f);

    /* Test Neural Network */
    printf("\nTesting Neural Network!\n");
    printf("Accuracy: %f %%\n", evaluate(testImages, testLabels));

    return EXIT_SUCCESS;
}