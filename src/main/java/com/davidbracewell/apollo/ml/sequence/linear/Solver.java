package com.davidbracewell.apollo.ml.sequence.linear;

/**
 * <p>Different solvers (optimizers) that can be used with the {@link CRFTrainer}</p>
 *
 * @author David B. Bracewell
 */
public enum Solver {
   /**
    * LBFGS solver
    */
   LBFGS("lbfgs"),
   /**
    * L2 regularized Stochastic Gradient Descent solver.
    */
   L2SGD("l2sgd"),
   /**
    * Average perceptron.
    */
   AVERAGE_PERCEPTRON("ap"),
   /**
    * Passive aggressive solver.
    */
   PASSIVE_AGGRESSIVE("pa"),
   /**
    * Adaptive regularization solver.
    */
   ADAPTIVE_REGULARIZATION("arow");

   final String parameterSetting;

   Solver(String parameterSetting) {
      this.parameterSetting = parameterSetting;
   }

}//END OF Solver
