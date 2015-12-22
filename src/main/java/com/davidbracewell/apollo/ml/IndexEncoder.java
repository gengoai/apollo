package com.davidbracewell.apollo.ml;

import com.davidbracewell.collection.Index;
import com.davidbracewell.collection.Indexes;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * An encoder backed by an <code>Index</code> allowing a finite number of objects to be mapped to double values.
 *
 * @author David B. Bracewell
 */
public class IndexEncoder implements Encoder, Serializable {
  private static final long serialVersionUID = 1L;
  private volatile Index<String> index = Indexes.newIndex();
  private volatile AtomicBoolean frozen = new AtomicBoolean(false);

  @Override
  public double get(Object object) {
    if (object == null) {
      return -1;
    }
    return index.indexOf(object.toString());
  }

  @Override
  public double encode(@NonNull Object object) {
    String str = object.toString();
    if (!frozen.get()) {
      return index.add(str);
    }
    return index.indexOf(str);
  }

  @Override
  public Object decode(double value) {
    return index.get((int) value);
  }

  @Override
  public void freeze() {
    frozen.set(true);
  }

  @Override
  public void unFreeze() {
    frozen.set(false);
  }

  @Override
  public boolean isFrozen() {
    return frozen.get();
  }

  @Override
  public int size() {
    return index.size();
  }

  @Override
  public List<Object> values() {
    return Collections.unmodifiableList(index.asList());
  }

  @Override
  public Encoder createNew() {
    return new IndexEncoder();
  }

}// END OF IndexEncoder
