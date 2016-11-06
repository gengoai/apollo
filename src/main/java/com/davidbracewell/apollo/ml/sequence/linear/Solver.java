package com.davidbracewell.apollo.ml.sequence.linear;

/**
 * @author David B. Bracewell
 */
public enum Solver {
   LBFGS("lbfgs"),
   L2SGD("l2sgd"),
   AVERAGE_PERCEPTRON("ap"),
   PASSIVE_AGGRESSIVE("pa"),
   ADAPTIVE_REGULARIZATION("arow");

   final String parameterSetting;

   Solver(String parameterSetting) {
      this.parameterSetting = parameterSetting;
   }
}//END OF Solver
