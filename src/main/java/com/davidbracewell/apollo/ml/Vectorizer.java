package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.function.SerializableFunction;

/**
 * @author David B. Bracewell
 */
public interface Vectorizer extends SerializableFunction<Example, Vector> {

   int getOutputDimension();

   void setEncoderPair(EncoderPair encoderPair);

}//END OF Vectorizer
