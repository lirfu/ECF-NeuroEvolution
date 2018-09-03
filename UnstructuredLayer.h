//
// Created by lirfu on 13.08.18..
//

#ifndef NEUROEVOLUTION_UNSTRUCTUREDLAYER_H
#define NEUROEVOLUTION_UNSTRUCTUREDLAYER_H

#include <memory>
#include <utility>
#include <bitset>
#include <functions/Linear.h>
#include "lib/nnpp/matrix/Matrix.h"
#include "lib/nnpp/layers/InnerLayer.h"
#include "lib/nnpp/functions/DerivativeFunction.h"

static uint index_ctr = 0;

class UnstructuredLayer : public InnerLayer<Matrix> {
public:
//    template<typename T>
//    class JoinedVector<T> {
//    private:
//        std::vector<std::vector<T> *> vectors_;
//        uint vectorIndex_ = 0;
//        uint innerIndex_ = 0;
//    public:
//        JoinedVector(initializer_list<std::vector<T> *> args) {
//            for (std::vector<T> *v : args)
//                vectors_.push_back(v);
//        }
//
//        bool hasNext() {
//            return vectorIndex_ < vectors_.size() && innerIndex_ < vectors_[vectorIndex_]->size();
//        }
//
//        T &next() {
//            T &t = vectors_[vectorIndex_]->at(innerIndex_);
//            innerIndex_++;
//            if (innerIndex_ >= vectors_[vectorIndex_]->size()) {
//                innerIndex_ = 0;
//                vectorIndex_++;
//            }
//            return t;
//        }
//    };

    class Neuron {
    private:
        DerivativeFunction *activation_;
        double net_;
        double biasDelta_;
        bool isInput_;
        bool isDone_;
    public:
        uint index;
        vector<Neuron *> predecessors_;
        vector<Neuron *> succesors_;
        vector<double> weights_;
        vector<double> weightDeltas_;
        double bias_;
        double output_;
        double grad_;

        explicit Neuron(DerivativeFunction *a, bool input = false) : activation_(a), index(index_ctr++) {
            isInput_ = input;
        }

        Neuron(Neuron &n) : activation_(n.activation_), bias_(n.bias_), weights_(n.weights_),
                            predecessors_(n.predecessors_), succesors_(n.succesors_), weightDeltas_(n.weightDeltas_),
                            index(index_ctr++) {}

        void reset() {
            isDone_ = false;
        }

        void forwardPass() {
            if (isDone_) return; // Against calculating twice.
            if (!isInput_) { // Inputs need not calculate outputs.
                net_ = bias_;
                uint i = 0;
                for (Neuron *n:predecessors_) {
                    n->forwardPass();// Recurse to inputs to get relevant outputs.
                    net_ += weights_[i++] * n->output_;
                }
                output_ = activation_->getFunction()(net_);
            }
            isDone_ = true;
        }

        void backwardPass(double learningRate) {
            if (isDone_) return;  // Against calculating twice.
            for (Neuron *n:succesors_) // Recurse to outputs to get relevant gradients.
                n->backwardPass(learningRate);
            // Inputs need not calculate gradients.
            if (!isInput_) {
                // Calculate gradients.
                double ds = grad_ * activation_->getDerivative()(net_);
                biasDelta_ -= learningRate * ds;
                for (uint i = 0; i < predecessors_.size(); i++) {
                    weightDeltas_[i] -= learningRate * ds * predecessors_[i]->output_;
                    predecessors_[i]->grad_ += ds * weights_[i];
                }
                grad_ = 0;// Reset gradient.
            }
            isDone_ = true;
        }

        void updateWeights(ulong batchSize) {
            bias_ += biasDelta_ / batchSize;
            biasDelta_ = 0; // Reset bias delta.
            for (uint i = 0; i < weights_.size(); i++) {
                weights_[i] += weightDeltas_[i] / batchSize;
                weightDeltas_[i] = 0;  // Reset weight deltas.
            }
        }

        std::string toString() {
            std::stringstream str;
            str << "Neuron" << index << std::endl;
            if (!isInput_) str << "b -> " << bias_ << std::endl;
            for (uint i = 0; i < predecessors_.size(); i++)
                str << predecessors_[i]->index << " -> " << setprecision(6) << weights_[i] << std::endl;
//            for (Neuron *n:succesors_)
//                str << n->index << ",";
//            str << endl;
            return str.str();
        }
    };

private:
    std::vector<Neuron *> inputNeurons_;
    std::vector<Neuron *> outputNeurons_;
    std::vector<Neuron *> hiddenNeurons_;
    Linear linearFunction_;
    std::shared_ptr<DerivativeFunction> hiddenFunction_;
    std::shared_ptr<DerivativeFunction> outputFunction_;
    bool skipWeightInit_;
protected:
    explicit UnstructuredLayer(Layer<Matrix> &layer) : InnerLayer(layer) {}

public:
    UnstructuredLayer(uint inputSize, uint outputSize,
                      std::shared_ptr<DerivativeFunction> hiddenFunction,
                      std::shared_ptr<DerivativeFunction> outputFunction, bool skipWeightInit = false) :
            InnerLayer<Matrix>(*new Matrix(1, outputSize)), hiddenFunction_(hiddenFunction),
            outputFunction_(outputFunction), skipWeightInit_(skipWeightInit) {
        // The ancestor neuron.
        Neuron *ancestor = new Neuron(hiddenFunction_.get());
        ancestor->index = 0;
        hiddenNeurons_.push_back(ancestor);
        // Build input neurons.
        index_ctr -= inputSize + outputSize + 1;
        for (uint i = 0; i < inputSize; i++) {
            Neuron *n = new Neuron(&linearFunction_, true);
            n->succesors_.push_back(ancestor);
            ancestor->predecessors_.push_back(n);
            ancestor->weights_.push_back(1);
            inputNeurons_.push_back(n);
        }
        // Build output neurons.
        for (uint i = 0; i < outputSize; i++) {
            Neuron *n = new Neuron(outputFunction_.get());
            ancestor->succesors_.push_back(n);
            n->predecessors_.push_back(ancestor);
            n->weights_.push_back(1);
            outputNeurons_.push_back(n);
        }
        index_ctr++;
    }

    ~UnstructuredLayer() {
        for (Neuron *n:inputNeurons_) delete n;
        for (Neuron *n:hiddenNeurons_) delete n;
        for (Neuron *n:outputNeurons_) delete n;
    }

    void initialize(WeightInitializer *initializer) override {
        if (!skipWeightInit_) {
            for (Neuron *n:hiddenNeurons_) {
                initializer->initialize(n->bias_);
                for (uint i = 0; i < n->weights_.size(); i++)
                    initializer->initialize(n->weights_[i]);
            }
            for (Neuron *n:outputNeurons_) {
                initializer->initialize(n->bias_);
                for (uint i = 0; i < n->weights_.size(); i++)
                    initializer->initialize(n->weights_[i]);
            }
        }
        uint i = 0;
        for (Neuron *n:hiddenNeurons_) {
            n->index = i++; // Correct neuron indexes to match those used with functions.
            for (uint j = 0; j < n->weights_.size(); j++)
                n->weightDeltas_.push_back(0);
//            n->weightDeltas_.resize(n->weights_.size());
        }
        for (Neuron *n:outputNeurons_) {
            for (uint j = 0; j < n->weights_.size(); j++)
                n->weightDeltas_.push_back(0);
//            n->weightDeltas_.resize(n->weights_.size());
        }
    }

    void forwardPass(Layer<Matrix> &leftLayer) override {
        Matrix &lo = leftLayer.getOutput();
        // Set inputs to input neurons.
        for (uint i = 0; i < inputNeurons_.size(); i++)
            inputNeurons_[i]->output_ = lo.get(0, i);
        // Recurse from outputs.
        for (Neuron *n:outputNeurons_) n->forwardPass();
        // Reset neurons.
        for (Neuron *n:inputNeurons_) n->reset();
        for (Neuron *n:hiddenNeurons_) n->reset();
        for (Neuron *n:outputNeurons_) n->reset();
        // Copy output neurons to output matrix.
        for (uint i = 0; i < outputNeurons_.size(); i++)
            this->output_.set(0, i, outputNeurons_[i]->output_);
    }

    Matrix &backwardPass(Matrix &outDiff, Matrix &leftOutputs, double learningRate) override {
        // Set gradients to output neurons.
        for (uint i = 0; i < outputNeurons_.size(); i++)
            outputNeurons_[i]->grad_ = outDiff.get(i, 0);
        // Recurse from inputs.
        for (Neuron *n:inputNeurons_) n->backwardPass(learningRate);
        // Reset neurons.
        for (Neuron *n:inputNeurons_) n->reset();
        for (Neuron *n:hiddenNeurons_) n->reset();
        for (Neuron *n:outputNeurons_) n->reset();
        // Set gradients for next matrix.
        outDiff = Matrix(inputNeurons_.size(), 1);
        for (uint i = 0; i < inputNeurons_.size(); i++)
            outDiff.set(i, 0, inputNeurons_[i]->grad_);
        return outDiff;
    }

    void updateWeights(ulong batchSize) override {
        for (Neuron *n:hiddenNeurons_) n->updateWeights(batchSize);
        for (Neuron *n:outputNeurons_) n->updateWeights(batchSize);
    }

    Neuron *parallelSplitNeuronAt(uint index) {
        Neuron &n = *hiddenNeurons_[index];
        ulong nextIndex = hiddenNeurons_.size();
        Neuron *newNeuron = new Neuron(n);
        for (Neuron *ne:n.predecessors_)// Notify predecesors.
            ne->succesors_.push_back(newNeuron);
        for (Neuron *ne:n.succesors_) { // Notify successors, add new pointer.
            ne->predecessors_.push_back(newNeuron);
            ne->weights_.push_back(1);
        }
        return newNeuron;
    }

    Neuron *serialSplitNeuronAt(uint index) {
        Neuron *n = hiddenNeurons_[index];
        Neuron *newNeuron = new Neuron(*n);
        for (Neuron *ne:n->succesors_) {// Notify successors, replace old pointer.
            index = static_cast<uint>(-1u);
            for (uint i = 0; i < ne->predecessors_.size(); i++)
                if (ne->predecessors_[i] == n) {
                    index = i;
                    break;
                }
            ne->predecessors_[index] = newNeuron;
        }
        n->succesors_.clear();
        n->succesors_.push_back(newNeuron);
        newNeuron->predecessors_.clear();
        newNeuron->predecessors_.push_back(n);
        newNeuron->weights_.clear();
        newNeuron->weights_.push_back(1);
        return newNeuron;
    }

    void addNeuron(Neuron *n) {
        hiddenNeurons_.push_back(n);
    }

    void cutConnection(uint neuron, uint connection_index) {
        Neuron *neuronN = hiddenNeurons_[neuron];
        vector<Neuron *> &suc = neuronN->predecessors_[connection_index]->succesors_;
        // Remove the link.
        neuronN->predecessors_.erase(neuronN->predecessors_.begin() + connection_index);
        neuronN->weights_.erase(neuronN->weights_.begin() + connection_index);
        // Remove predecessor's link.
        ulong index = 0;
        for (Neuron *n:suc) {
            if (n == neuronN) break;
            index++;
        }
        suc.erase(suc.begin() + index);
    }

    /*!
     * Get the pointer to a weight in the network.
     * @param neuron Index of the neuron.
     * @param weight_index Index of the weight for the neuron. If index is larger than the weights vector, the
     * position is calculated as: @code weight_index % weights_vector.size() @endcode If weight_index is -1u, bias is
     * returned.
     */
    double *getNeuronWeight(uint neuron, uint weight_index) {
        if (weight_index == -1u) {
            return &hiddenNeurons_[neuron]->bias_;
        }
        vector<double> &weights = hiddenNeurons_[neuron]->weights_;
        if (weight_index > weights.size()) weight_index = weight_index % (uint) weights.size();
        return &weights[weight_index];
    }

    uint numberOfParameters() override {
        uint num = 0;
        for (Neuron *n:hiddenNeurons_)
            num += n->weights_.size() + 1;
        for (Neuron *n:outputNeurons_)
            num += n->weights_.size() + 1;
        return num;
    }

    uint getNeuronNumber() override {
        return static_cast<uint>(hiddenNeurons_.size() + outputNeurons_.size());
    }

    void getNeuron(uint index, double *values) override {
        Neuron *n;
        if (index < hiddenNeurons_.size()) { // Output neuron
            n = hiddenNeurons_[index];
        } else {
            n = outputNeurons_[index - hiddenNeurons_.size()];
        }
        values = new double[n->weights_.size() + 1];
        uint i = 0;
        values[i++] = n->bias_;
        for (double v : n->weights_)
            values[i++] = v;
    }

    void setNeuron(uint index, const double *values) override {
        Neuron *n;
        if (index < hiddenNeurons_.size()) { // Output neuron
            n = hiddenNeurons_[index];
        } else {
            n = outputNeurons_[index - hiddenNeurons_.size()];
        }
        uint i = 0;
        n->bias_ = values[i++];
        for (; i <= n->weights_.size(); i++)
            n->weights_[i - 1] = values[i];
    }

    Layer<Matrix> *copy() override {
//        UnstructuredLayer l(inputNeurons_.size(), outputNeurons_.size(), hiddenFunction_, outputFunction_);
//        for(uint i=0; i<hiddenNeurons_.size(); i++)
//            l.hiddenNeurons_.push_back(new Neuron(*hiddenNeurons_[i]));
//        for(uint i=0; i<outputNeurons_.size(); i++)
//            l.outputNeurons_[i](new Neuron(*hiddenNeurons_[i]))
        return nullptr; // TODO
    }

    std::string toString() override {
        std::stringstream str;
        str << "> Inputs:" << endl;
        for (Neuron *n:inputNeurons_)
            str << n->toString();
        str << "> Hiddens:" << endl;
        for (Neuron *n:hiddenNeurons_)
            str << n->toString();
        str << "> Outputs:" << endl;
        for (Neuron *n:outputNeurons_)
            str << n->toString();
        return str.str();
    }
};

#endif //NEUROEVOLUTION_UNSTRUCTUREDLAYER_H
