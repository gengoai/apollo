package com.gengoai.apollo.ml.regression;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import de.bwaldvogel.liblinear.*;
import lombok.Getter;
import lombok.Setter;

import java.util.Iterator;
import java.util.List;

import static com.gengoai.collection.Lists.asArrayList;


/**
 * <p>Trains a regression model using LibLinear (L2R_L2LOSS_SVR)</p>
 *
 * @author David B. Bracewell
 */
public class LibLinearRegression extends RegressionLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private double C = 1;
   @Getter
   @Setter
   private double eps = 0.0001;
   @Getter
   @Setter
   private boolean verbose = false;

   private static Feature[] toFeature(NDArray vector, int biasIndex) {
      List<NDArray.Entry> entries = asArrayList(vector.sparseOrderedIterator());
      Feature[] feature = new Feature[entries.size() + 1];
      for (int i = 0; i < entries.size(); i++) {
         feature[i] = new FeatureNode(entries.get(i).matrixIndex() + 1, entries.get(i).getValue());
      }
      feature[entries.size()] = new FeatureNode(biasIndex, 1.0);
      return feature;
   }

   @Override
   public void resetLearnerParameters() {

   }

   @Override
   protected Regression trainImpl(Dataset<Instance> dataset) {
      Problem problem = new Problem();
      problem.l = dataset.size();
      problem.x = new Feature[problem.l][];
      problem.y = new double[problem.l];
      problem.bias = 0;

      int index = 0;
      int biasIndex = dataset.getFeatureEncoder().size() + 1;

      for (Iterator<Instance> iitr = dataset.iterator(); iitr.hasNext(); index++) {
         NDArray vector = iitr.next().toVector(dataset.getEncoderPair());
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

      SimpleRegressionModel srm = new SimpleRegressionModel(this);

      double[] modelWeights = model.getFeatureWeights();
      srm.weights = NDArrayFactory.DEFAULT().zeros(srm.numberOfFeatures());
      for (int i = 0; i < srm.numberOfFeatures(); i++) {
         srm.weights.set(i, modelWeights[i]);
      }
      srm.bias = modelWeights.length > model.getNrFeature() ? modelWeights[model.getNrFeature()] : 0d;
      return srm;
   }

}// END OF LibLinearRegression
