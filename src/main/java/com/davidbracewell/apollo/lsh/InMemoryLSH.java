package com.davidbracewell.apollo.lsh;


import com.davidbracewell.apollo.linalg.Vector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * The type In memory lSH.
 *
 * @param <V> the type parameter
 * @author dbracewell
 */
public class InMemoryLSH<V extends Vector> extends LSH<V> {
  private static final long serialVersionUID = 1L;

  private List<V> vectors = new ArrayList<>();

  /**
   * Instantiates a new In memory lSH.
   *
   * @param hashFamily      the hash family
   * @param numberOfHashes  the number of hashes
   * @param numberOfBuckets the number of buckets
   */
  public InMemoryLSH(HashFamily hashFamily, int numberOfHashes, int numberOfBuckets) {
    super(hashFamily, createTables(numberOfHashes, numberOfBuckets, hashFamily));
  }

  private static LSHTable[] createTables(int numberOfHashes, int numberOfBuckets, HashFamily family) {
    LSHTable[] tables = new LSHTable[numberOfBuckets];
    for (int i = 0; i < numberOfBuckets; i++) {
      tables[i] = new InMemoryLSHTable(new HashGroup(family, numberOfHashes));
    }
    return tables;
  }

  @Override
  protected int addVector(V vector) {
    int index = vectors.size();
    vectors.add(vector);
    return index;
  }

  @Override
  public void close() throws IOException {

  }

  @Override
  public void commit() {

  }

  @Override
  protected V getVector(int index) {
    return vectors.get(index);
  }

}//END OF InMemoryLSH
