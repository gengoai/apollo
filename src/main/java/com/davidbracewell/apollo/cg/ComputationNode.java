package com.davidbracewell.apollo.cg;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public interface ComputationNode extends Serializable {

   NDArray getOutput();

   void setOutput(NDArray output);

   default void setOutput(double value) {
      setOutput(NDArrayFactory.DENSE_DOUBLE.scalar(value));
   }

}//END OF ComputationNode
