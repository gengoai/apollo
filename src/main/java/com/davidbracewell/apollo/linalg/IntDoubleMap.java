package com.davidbracewell.apollo.linalg;

import lombok.Value;

import java.io.Serializable;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * @author David B. Bracewell
 */
public class IntDoubleMap implements Serializable, Iterable<IntDoubleMap.Entry> {
  private static final int DEFAULT_CAPACITY = 4;
  private int size;
  private int tableSize;
  private int[] keys;
  private double[] values;
  private byte[] state;

  public IntDoubleMap() {
    this(DEFAULT_CAPACITY);
  }

  public IntDoubleMap(int capacity) {
    this.tableSize = capacity;
    this.keys = new int[capacity];
    this.values = new double[capacity];
    this.state = new byte[capacity];
    this.size = 0;
  }

  public int size() {
    return size;
  }

  public boolean containsKey(int key) {
    if (size == 0) {
      return false;
    }
    return get(key) == 0;
  }

  public double get(int key) {
    if (size == 0) {
      return 0d;
    }
    int position;
    for (position = hash(key); state[position] == 1; position = (position + 1) % tableSize) {
      if (keys[position] == key) {
        return values[position];
      }
    }
    return 0d;
  }

  public double put(int key, double value) {
    if (size >= tableSize / 2) {
      resize(tableSize * 2);
    }

    if (value == 0) {
      return removeKey(key);
    }

    int position;
    for (position = hash(key); state[position] == 1; position = (position + 1) % tableSize) {
      if (keys[position] == key) {
        double old = values[position];
        values[position] = value;
        return old;
      }
    }
    state[position] = 1;
    keys[position] = key;
    values[position] = value;
    size++;
    return 0d;
  }

  public double removeKey(int key) {
    int position;

    if (!containsKey(key)) {
      return 0d;
    }

    double old = 0;
    for (position = hash(key); state[position] == 1; position = (position + 1) % tableSize) {
      if (keys[position] == key) {
        old = values[position];
        state[position] = 0;
        keys[position] = 0;
        values[position] = 0;
        size--;
        break;
      }
    }

    position = (position + 1) % tableSize;
    while (state[position] != 1) {
      int k = keys[position];
      double v = values[position];
      state[position] = 0;
      keys[position] = 0;
      values[position] = 0;
      put(k, v);
      position = (position + 1) % tableSize;
    }

    return old;
  }

  private int hash(int key) {
    return (Integer.hashCode(key) & 0x7fffffff) % tableSize;
  }

  private void resize(int newSize) {
    IntDoubleMap map = new IntDoubleMap(newSize);
    for (int i = 0; i < tableSize; i++) {
      if (state[i] == 1) {
        map.put(keys[i], values[i]);
      }
    }
    this.size = map.size;
    this.state = map.state;
    this.keys = map.keys;
    this.values = map.values;
    this.tableSize = map.tableSize;
  }

  @Override
  public Iterator<Entry> iterator() {
    return new Iterator<Entry>() {
      private int position = -1;
      private Integer key = null;
      private double value;

      private boolean advance() {
        if (key == null) {
          for (position = position + 1; position < tableSize; position++) {
            if (state[position] == 1) {
              key = keys[position];
              value = values[position];
              return true;
            }
          }
          key = null;
        }
        return key != null;
      }

      @Override
      public boolean hasNext() {
        return advance();
      }

      @Override
      public Entry next() {
        if (!advance()) {
          throw new NoSuchElementException();
        }
        Entry entry = new Entry(key, value);
        key = null;
        return entry;
      }
    };
  }

  private int[] keys() {
    int[] a = new int[size];
    for (int i = 0, j = 0; i < tableSize & j < size; i++) {
      if (state[i] == 1) {
        a[j] = keys[i];
        j++;
      }
    }
    return a;
  }

  private double[] values() {
    double[] a = new double[size];
    for (int i = 0, j = 0; i < tableSize & j < size; i++) {
      if (state[i] == 1) {
        a[j] = values[i];
        j++;
      }
    }
    return a;
  }

  public void clear() {
    this.size = 0;
    this.tableSize = DEFAULT_CAPACITY;
    this.state = new byte[DEFAULT_CAPACITY];
    this.keys = new int[DEFAULT_CAPACITY];
    this.values = new double[DEFAULT_CAPACITY];
  }

  @Value
  public static class Entry {
    private final int key;
    private final double value;

    @Override
    public String toString() {
      return "(" + key + ", " + value + ")";
    }

  }


}// END OF IntDoubleMap

