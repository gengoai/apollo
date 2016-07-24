package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.collection.HashMapIndex;
import com.davidbracewell.collection.Index;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.stream.accumulator.CollectionAccumulatable;
import com.davidbracewell.stream.accumulator.MAccumulator;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * An encoder backed by an <code>Index</code> allowing a finite number of objects to be mapped to double values.
 *
 * @author David B. Bracewell
 */
public class IndexEncoder implements Encoder, Serializable {
  private static final long serialVersionUID = 1L;
  private volatile Index<String> index = new HashMapIndex<>();
  private volatile AtomicBoolean frozen = new AtomicBoolean(false);

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
      MAccumulator<Set<String>> accumulator = dataset.getStreamingContext().accumulator(new HashSet<>(), new CollectionAccumulatable<>());
      dataset.stream().forEach(ex -> accumulator.add(ex.getFeatureSpace().collect(Collectors.toSet())));
      this.index.addAll(accumulator.value());
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
