package com.davidbracewell.apollo.cg;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;

/**
 * @author David B. Bracewell
 */
public class Variable implements ComputationNode {
   private static final long serialVersionUID = 1L;
   private NDArray value;

   public Variable() {
      value = NDArrayFactory.DENSE_DOUBLE.empty();
   }

   public Variable(double value) {
      setOutput(value);
   }

   public Variable(NDArray value) {
      setOutput(value);
   }

   @Override
   public NDArray getOutput() {
      return value;
   }

   @Override
   public void setOutput(NDArray output) {
      if (output == null) {
         value = NDArrayFactory.DENSE_DOUBLE.empty();
      } else {
         value = output;
      }
   }
}// END OF Variable
