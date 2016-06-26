package com.davidbracewell.apollo.linalg;

import com.google.common.base.Preconditions;
import lombok.NonNull;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.util.function.BiFunction;

/**
 * The type Lsh.
 *
 * @author David B. Bracewell
 */
public class InMemoryLSH extends LSH {
  private static final long serialVersionUID = 1L;
  private OpenIntHashSet[][] vectorStore;


  /**
   * Instantiates a new Lsh.
   *
   * @param bands             the bands
   * @param buckets           the buckets
   * @param signatureFunction the signature function
   */
  public InMemoryLSH(int bands, int buckets, @NonNull SignatureFunction signatureFunction) {
    super(bands, buckets, signatureFunction);
    initVectorStore();
  }

  /**
   * Instantiates a new Lsh.
   *
   * @param bands             the bands
   * @param buckets           the buckets
   * @param dimension         the dimension
   * @param threshold         the threshold
   * @param signatureSupplier the signature supplier
   */
  public InMemoryLSH(int bands, int buckets, int dimension, double threshold, @NonNull BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
    super(bands, buckets, dimension, threshold, signatureSupplier);
    initVectorStore();
  }

  /**
   * Instantiates a new Lsh.
   *
   * @param bands             the bands
   * @param buckets           the buckets
   * @param dimension         the dimension
   * @param signatureSupplier the signature supplier
   */
  public InMemoryLSH(int bands, int buckets, int dimension, @NonNull BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
    this(bands, buckets, dimension, 0.5, signatureSupplier);
  }

  public static Builder builder() {
    return new Builder();
  }

  private void initVectorStore() {
    this.vectorStore = new OpenIntHashSet[getBands()][getBuckets()];
    for (int b = 0; b < getBands(); b++) {
      for (int u = 0; u < getBuckets(); u++) {
        this.vectorStore[b][u] = new OpenIntHashSet();
      }
    }
  }

  @Override
  public void clear() {
    initVectorStore();
  }

  @Override
  protected OpenIntHashSet get(int band, int bucket) {
    return vectorStore[band][bucket];
  }

  @Override
  protected void addToTable(int band, int bucket, int vid) {
    vectorStore[band][bucket].add(vid);
  }

  public static class Builder extends LSH.Builder {

    @Override
    public InMemoryLSH create() {
      if (signatureFunction != null) {
        return new InMemoryLSH(bands, buckets, signatureFunction);
      }
      Preconditions.checkArgument(dimension > 0, "Dimension not set.");
      Preconditions.checkNotNull(signatureSupplier, "A signature suppler was not set");
      return new InMemoryLSH(bands, buckets, dimension, threshold, signatureSupplier);
    }

    @Override
    public <KEY> VectorStore<KEY> createVectorStore(){
      return new InMemoryLSHVectorStore<>(create());
    }

  }

}// END OF LSH