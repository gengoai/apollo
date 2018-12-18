package com.gengoai.apollo.ml.regression;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.LibLinear;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;

/**
 * <p>Regression model using LibLinear</p>
 *
 * @author David B. Bracewell
 */
public class LibLinearLinearRegression implements Regression {
   private static final long serialVersionUID = 1L;
   private Model model;
   private int biasIndex;

   @Override
   public NDArray estimate(NDArray data) {
      return LibLinear.estimate(model, data, biasIndex);
   }

   /**
    * Specialized fit method that takes a Parameters object.
    *
    * @param dataSupplier  the data supplier
    * @param fitParameters the fit parameters
    */
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      biasIndex = (fitParameters.bias ? fitParameters.numFeatures + 1 : -1);
      model = LibLinear.fit(dataSupplier,
                            new Parameter(SolverType.L2R_L2LOSS_SVR,
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
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters fitParameters) {
      fit(dataSupplier, Cast.as(fitParameters, Parameters.class));
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return new Parameters();
   }

   @Override
   public int getNumberOfFeatures() {
      return model.getNrFeature();
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
       * The maximum number of iterations to run the trainer (Default 1000)
       */
      public int maxIterations = 1000;

      /**
       * The epsilon in loss function of epsilon-SVR (default 0.1)
       */
      public double p = 0.1;

   }

}//END OF MultivariateLinearRegression
