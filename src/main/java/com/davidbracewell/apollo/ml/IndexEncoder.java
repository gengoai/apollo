package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.collection.HashMapIndex;
import com.davidbracewell.collection.Index;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * An encoder backed by an <code>Index</code> allowing a finite number of objects to be mapped to double values.
 *
 * @author David B. Bracewell
 */
public class IndexEncoder implements Encoder, Serializable {
  private static final long serialVersionUID = 1L;
  protected volatile Index<String> index = new HashMapIndex<>();
  protected volatile AtomicBoolean frozen = new AtomicBoolean(false);

  @Override
  public double get(Object object) {
    if (object == null) {
      return -1;
    } else if (object instanceof Collection) {
      Collection<?> collection = Cast.as(object);
      double idx = -1;
      for (Object o : collection) {
        idx = index.indexOf(o.toString());
      }
      return idx;
    }
    return index.indexOf(object.toString());
  }

  @Override
  public void fit(@NonNull Dataset<? extends Example> dataset) {
    if (!isFrozen()) {
      this.index.addAll(
        dataset.stream()
//          .parallel()
          .flatMap(ex -> ex.getFeatureSpace().filter(Objects::nonNull).collect(Collectors.toSet()))
          .filter(Objects::nonNull)
          .distinct()
          .collect()
      );
    }
  }

  @Override
  public double encode(Object object) {
    if (object == null) {
      return -1;
    }
    if (object instanceof Collection) {
      Collection<?> collection = Cast.as(object);
      double idx = -1;
      for (Object o : collection) {
        if (!frozen.get()) {
          idx = index.add(o.toString());
        } else {
          idx = index.indexOf(o.toString());
        }
      }
      return idx;
    }
    String str = object.toString();
    if (str != null) {
      if (!frozen.get()) {
        return index.add(str);
      }
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
