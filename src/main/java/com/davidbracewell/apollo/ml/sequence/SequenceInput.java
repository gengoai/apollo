package com.davidbracewell.apollo.ml.sequence;

import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Optional;

/**
 * @author David B. Bracewell
 */
public class SequenceInput<T> implements Serializable {
  private static final long serialVersionUID = 1L;
  private final ArrayList<T> list = new ArrayList<>();
  private final ArrayList<String> labels = new ArrayList<>();

  public SequenceInput() {

  }

  public SequenceInput(@NonNull Collection<T> collection) {
    list.addAll(collection);
  }

  public String getLabel(int index) {
    if (index < 0 || index >= labels.size()) {
      return null;
    }
    return labels.get(index);
  }

  public T get(int index) {
    if (index < 0 || index >= list.size()) {
      return null;
    }
    return list.get(index);
  }


  public void add(T observation, String label) {
    list.add(observation);
    labels.add(label);
  }

  public void add(T observation) {
    list.add(observation);
    labels.add(null);
  }

  public int size() {
    return list.size();
  }

  public ContextualIterator<T> iterator() {
    return new Itr();
  }


  private class Itr extends ContextualIterator<T> {
    private static final long serialVersionUID = 1L;

    @Override
    public int size() {
      return SequenceInput.this.size();
    }

    @Override
    protected Optional<T> getContextAt(int index) {
      return Optional.ofNullable(SequenceInput.this.get(index));
    }

    @Override
    protected Optional<String> getLabelAt(int index) {
      return Optional.ofNullable(SequenceInput.this.getLabel(index));
    }
  }


}// END OF SequenceInput
