package com.davidbracewell.apollo.ml;

import com.davidbracewell.collection.Index;
import com.davidbracewell.collection.Indexes;

import java.io.Serializable;
import java.util.Collection;

/**
 * @author David B. Bracewell
 */
public class DiscreteLabelEncoder implements LabelEncoder, Serializable {
  private static final long serialVersionUID = 1L;
  private final Index<String> index = Indexes.newIndex();

  @Override
  public double encode(Object object) {
    return index.add(object.toString());
  }

  @Override
  public Object decode(double value) {
    return index.get((int) value);
  }

  @Override
  public Collection<String> labels() {
    return index.asList();
  }

}// END OF DiscreteLabelEncoder
