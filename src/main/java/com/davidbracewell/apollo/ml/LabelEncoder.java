package com.davidbracewell.apollo.ml;

import java.util.Collection;

/**
 * @author David B. Bracewell
 */
public interface LabelEncoder {

  static LabelEncoder discrete() {
    return new DiscreteLabelEncoder();
  }

  double encode(Object object);

  Object decode(double value);

  Collection<String> labels();


}// END OF LabelEncoder
