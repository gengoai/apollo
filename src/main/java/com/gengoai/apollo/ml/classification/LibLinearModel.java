package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.LibLinear;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.logging.Loggable;
import com.gengoai.stream.MStream;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;

/**
 * The type Lib linear model.
 *
 * @author David B. Bracewell
 */
public class LibLinearModel implements Classifier, Loggable {
   private static final long serialVersionUID = 1L;
   private int biasIndex = -1;
   private Model model;

   /**
    * Specialized fit method taking the LibLinear Parameters object.
    *
    * @param dataSupplier  the data supplier
    * @param fitParameters the fit parameters
    */
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      biasIndex = (fitParameters.bias ? fitParameters.numFeatures + 1 : -1);
      model = LibLinear.fit(dataSupplier,
                            new Parameter(fitParameters.solver,
                                          fitParameters.C,
                                          fitParameters.eps,
                                          fitParameters.maxIterations,
                                          fitParameters.p),
                            fitParameters.verbose,
                            fitParameters.numFeatures,
                            biasIndex
                           );
   }

   @Override
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters parameters) {
      fit(dataSupplier, Cast.as(parameters, Parameters.class));
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   @Override
   public NDArray estimate(NDArray data) {
      return LibLinear.estimate(model, data, biasIndex);
   }

   @Override
   public int getNumberOfFeatures() {
      return model.getNrFeature();
   }

   @Override
   public int getNumberOfLabels() {
      return model.getNrClass();
   }

   /**
    * Custom fit parameters for LibLinear
    */
   public static class Parameters extends FitParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The cost parameter (default 1.0)
       */
      public double C = 1.0;
      /**
       * Use a bias feature or not. (default false)
       */
      public boolean bias = false;
      /**
       * The tolerance for termination.(default 0.0001)
       */
      public double eps = 0.0001;
      /**
       * The Solver to use. (default L2R_LR)
       */
      public SolverType solver = SolverType.L2R_LR;
      /**
       * The maximum number of iterations to run the trainer (Default 1000)
       */
      public int maxIterations = 1000;

      /**
       * The epsilon in loss function of epsilon-SVR (default 0.1)
       */
      public double p = 0.1;

   }
}//END OF LibLinearModel
