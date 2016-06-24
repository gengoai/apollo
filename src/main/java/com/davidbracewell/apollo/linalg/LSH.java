package com.davidbracewell.apollo.linalg;

import com.davidbracewell.apollo.affinity.Measure;
import com.davidbracewell.apollo.affinity.Optimum;
import lombok.Getter;
import lombok.NonNull;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.io.Serializable;
import java.util.function.BiFunction;

/**
 * The type Lsh.
 *
 * @author David B. Bracewell
 */
public abstract class LSH implements Serializable {
  private static final long LARGE_PRIME = 433494437;

  @Getter
  private final int bands;
  @Getter
  private final int buckets;
  @Getter
  private final int dimension;
  @Getter
  private final SignatureFunction signatureFunction;


  /**
   * Instantiates a new Lsh.
   *
   * @param bands             the bands
   * @param buckets           the buckets
   * @param signatureFunction the signature function
   */
  public LSH(int bands, int buckets, @NonNull SignatureFunction signatureFunction) {
    this.bands = bands;
    this.buckets = buckets;
    this.dimension = signatureFunction.getDimension();
    this.signatureFunction = signatureFunction;
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
  public LSH(int bands, int buckets, int dimension, double threshold, @NonNull BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
    this.bands = bands;
    this.buckets = buckets;
    this.dimension = dimension;
    int r = (int) (Math.ceil(Math.log(1.0 / bands) / Math.log(threshold)) + 1);
    int signature_size = r * bands;
    this.signatureFunction = signatureSupplier.apply(signature_size, dimension);
  }

  /**
   * Instantiates a new Lsh.
   *
   * @param bands             the bands
   * @param buckets           the buckets
   * @param dimension         the dimension
   * @param signatureSupplier the signature supplier
   */
  public LSH(int bands, int buckets, int dimension, @NonNull BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
    this(bands, buckets, dimension, 0.5, signatureSupplier);
  }


  /**
   * Clear.
   */
  public abstract void clear();

  /**
   * Get open int hash set.
   *
   * @param band   the band
   * @param bucket the bucket
   * @return the open int hash set
   */
  protected abstract OpenIntHashSet get(int band, int bucket);

  /**
   * Neighbors list.
   *
   * @param vector the vector
   * @return the list
   */
  public OpenIntHashSet query(@NonNull Vector vector) {
    OpenIntHashSet matches = new OpenIntHashSet();
    int[] hash = hash(vector);
    for (int i = 0; i < bands; i++) {
      get(i, hash[i]).forEachKey(matches::add);
    }
    return matches;
  }

  /**
   * Add to table.
   *
   * @param band   the band
   * @param bucket the bucket
   * @param vid    the vid
   */
  protected abstract void addToTable(int band, int bucket, int vid);

  /**
   * Add.
   *
   * @param vector   the vector
   * @param vectorID the vector id
   */
  public void add(@NonNull Vector vector, int vectorID) {
    int[] hash = hash(vector);
    for (int band = 0; band < bands; band++) {
      addToTable(band, hash[band], vectorID);
    }
  }

  /**
   * Remove.
   *
   * @param vector   the vector
   * @param vectorID the vector id
   */
  public void remove(Vector vector, int vectorID) {
    if (vector != null) {
      int[] hash = hash(vector);
      for (int band = 0; band < bands; band++) {
        get(band, hash[band]).remove(vectorID);
      }
    }
  }

  /**
   * Remove.
   *
   * @param vectorID the vector id
   */
  public void remove(int vectorID) {
    for (int band = 0; band < bands; band++) {
      for (int bucket = 0; bucket < buckets; bucket++) {
        get(band, bucket).remove(vectorID);
      }
    }
  }

  private int[] hash(Vector vector) {
    if (signatureFunction.isBinary()) {
      return booleanSignatureHash(signatureFunction.signature(vector));
    }
    return intSignatureHash(signatureFunction.signature(vector));
  }

  private int[] intSignatureHash(final int[] signature) {
    int[] hash = new int[bands];
    int rows = signature.length / bands;
    for (int index = 0; index < signature.length; index++) {
      int band = Math.min(index / rows, bands - 1);
      hash[band] = (int) ((hash[band] + (long) signature[index] * LARGE_PRIME) % buckets);
    }
    return hash;
  }

  private int[] booleanSignatureHash(final int[] signature) {
    long[] acc = new long[bands];
    int rows = signature.length / bands;
    for (int index = 0; index < signature.length; index++) {
      long v = 0;
      if (signature[index] == 1) {
        v = (index + 1) * LARGE_PRIME;
      }
      int j = Math.min(index / rows, bands - 1);
      acc[j] = (acc[j] + v) % Integer.MAX_VALUE;
    }

    int[] hash = new int[bands];
    for (int i = 0; i < bands; i++) {
      hash[i] = (int) (acc[i] % buckets);
    }
    return hash;
  }


  /**
   * Gets measure.
   *
   * @return the measure
   */
  public Measure getMeasure() {
    return signatureFunction.getMeasure();
  }

  /**
   * Gets optimum.
   *
   * @return the optimum
   */
  public Optimum getOptimum() {
    return signatureFunction.getMeasure().getOptimum();
  }

  protected abstract static class Builder {
    protected int bands = 5;
    protected int buckets = 20;
    protected double threshold = 0.5;
    protected int dimension = -1;
    protected SignatureFunction signatureFunction = null;
    protected BiFunction<Integer, Integer, SignatureFunction> signatureSupplier = CosineSignature::new;

    public Builder bands(int bands) {
      this.bands = bands;
      return this;
    }

    public Builder buckets(int buckets) {
      this.buckets = buckets;
      return this;
    }

    public Builder dimension(int dimension) {
      this.dimension = dimension;
      return this;
    }

    public Builder threshiold(double threshold) {
      this.threshold = threshold;
      return this;
    }

    public Builder signatureFunction(SignatureFunction signatureFunction) {
      this.signatureFunction = signatureFunction;
      return this;
    }

    public Builder signatureSupplier(BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
      this.signatureSupplier = signatureSupplier;
      return this;
    }

    public abstract LSH create();

    public abstract <KEY> VectorStore<KEY> createVectorStore();

  }


}// END OF LSH
