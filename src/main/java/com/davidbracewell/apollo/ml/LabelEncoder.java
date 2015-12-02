package com.davidbracewell.apollo.ml;

import com.davidbracewell.reflection.Reflect;
import com.davidbracewell.reflection.ReflectionException;
import com.google.common.base.Throwables;

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

  default LabelEncoder createNew() {
    try {
      return Reflect.onClass(getClass()).create().get();
    } catch (ReflectionException e) {
      throw Throwables.propagate(e);
    }
  }


}// END OF LabelEncoder
