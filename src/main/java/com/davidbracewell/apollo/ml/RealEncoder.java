package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.conversion.Cast;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

/**
 * An encoder that expects to encode Numbers to values. If non-numbers are given, and
 * <code>IllegalArgumentException</code> is thrown.
 *
 * @author David B. Bracewell
 */
public class RealEncoder implements Serializable, LabelEncoder {
  private static final long serialVersionUID = 1L;

  @Override
  public double get(Object object) {
    return encode(object);
  }

  @Override
  public void fit(Dataset<? extends Example> dataset) {

  }

  @Override
  public double encode(@NonNull Object object) {
    Preconditions.checkArgument(object instanceof Number, object.getClass() + " is not a valid Number");
    return Cast.<Number>as(object).doubleValue();
  }

  @Override
  public Object decode(double value) {
    return value;
  }

  @Override
  public void freeze() {

  }

  @Override
  public void unFreeze() {

  }

  @Override
  public boolean isFrozen() {
    return true;
  }

  @Override
  public int size() {
    return 0;
  }

  @Override
  public List<Object> values() {
    return Collections.emptyList();
  }

  @Override
  public LabelEncoder createNew() {
    return new RealEncoder();
  }

}// END OF RealEncoder
