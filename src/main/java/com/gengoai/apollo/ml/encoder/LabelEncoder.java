package com.gengoai.apollo.ml.encoder;

/**
 * <p>Specialized encoders for class labels.</p>
 *
 * @author David B. Bracewell
 */
public interface LabelEncoder extends Encoder {

   @Override
   LabelEncoder createNew();

}//END OF LabelEncoder
