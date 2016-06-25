package com.davidbracewell.apollo.ml.regression;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.google.common.collect.Lists;
import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

import java.util.Iterator;
import java.util.List;

/**
 * The type Lib linear regression.
 *
 * @author David B. Bracewell
 */
public class LibLinearRegression extends RegressionLearner {
  private static final long serialVersionUID = 1L;
  private double C = 1;
  private double eps = 0.0001;
  private boolean verbose = false;

  /**
   * To feature feature [ ].
   *
   * @param vector    the vector
   * @param biasIndex the bias index
   * @return the feature [ ]
   */
  static Feature[] toFeature(Vector vector, int biasIndex) {
    List<Vector.Entry> entries = Lists.newArrayList(vector.orderedNonZeroIterator());
    Feature[] feature = new Feature[entries.size() + 1];
    for (int i = 0; i < entries.size(); i++) {
      feature[i] = new FeatureNode(entries.get(i).index + 1, entries.get(i).value);
    }
    feature[entries.size()] = new FeatureNode(biasIndex, 1.0);
    return feature;
  }

  @Override
  protected Regression trainImpl(Dataset<Instance> dataset) {
    Problem problem = new Problem();
    problem.l = dataset.size();
    problem.x = new Feature[problem.l][];
    problem.y = new double[problem.l];
    problem.bias = 0;

    int index = 0;
    int biasIndex = dataset.getFeatureEncoder().size()+1;

    for (Iterator<Instance> iitr = dataset.iterator(); iitr.hasNext(); index++) {
      FeatureVector vector = iitr.next().toVector(dataset.getEncoderPair());
      problem.x[index] = toFeature(vector, biasIndex);
      problem.y[index] = vector.getLabel();
    }
    problem.n = dataset.getFeatureEncoder().size() + 1;

    if (verbose) {
      Linear.enableDebugOutput();
    } else {
      Linear.disableDebugOutput();
    }

    Model model = Linear.train(problem, new Parameter(SolverType.L2R_L2LOSS_SVR, C, eps));

    SimpleRegressionModel srm = new SimpleRegressionModel(
      dataset.getEncoderPair(),
      dataset.getPreprocessors().getModelProcessors()
    );

    double[] modelWeights = model.getFeatureWeights();
    srm.weights = new SparseVector(srm.numberOfFeatures());
    for (int i = 0; i < srm.numberOfFeatures(); i++) {
      srm.weights.set(i, modelWeights[i]);
    }
    srm.bias = modelWeights.length > model.getNrFeature() ? modelWeights[model.getNrFeature()] : 0d;
    return srm;
  }

  @Override
  public void reset() {

  }


  /**
   * Gets c.
   *
   * @return the c
   */
  public double getC() {
    return C;
  }

  /**
   * Sets c.
   *
   * @param c the c
   */
  public void setC(double c) {
    C = c;
  }

  /**
   * Gets eps.
   *
   * @return the eps
   */
  public double getEps() {
    return eps;
  }

  /**
   * Sets eps.
   *
   * @param eps the eps
   */
  public void setEps(double eps) {
    this.eps = eps;
  }

  /**
   * Is verbose boolean.
   *
   * @return the boolean
   */
  public boolean isVerbose() {
    return verbose;
  }

  /**
   * Sets verbose.
   *
   * @param verbose the verbose
   */
  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }
}// END OF LibLinearRegression
