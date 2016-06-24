package com.davidbracewell.apollo.linalg;

import com.davidbracewell.string.StringUtils;
import com.google.common.base.Stopwatch;
import lombok.NonNull;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Lsh vector store.
 *
 * @param <KEY> the type parameter
 * @author David B. Bracewell
 */
public class InMemoryLSHVectorStore<KEY> extends LSHVectorStore<KEY> {
  private final AtomicInteger vectorIDGenerator = new AtomicInteger();
  private final OpenIntObjectHashMap<LabeledVector> vectorIDMap = new OpenIntObjectHashMap<>();
  private final OpenObjectIntHashMap<KEY> keys = new OpenObjectIntHashMap<>();


  /**
   * Instantiates a new Lsh vector store.
   *
   * @param lsh the lsh
   */
  public InMemoryLSHVectorStore(@NonNull InMemoryLSH lsh) {
    super(lsh);
  }

  /**
   * The entry point of application.
   *
   * @param args the input arguments
   */
  public static void main(String[] args) {
    VectorStore<String> vectorStore = InMemoryLSH.builder()
      .dimension(100)
      .signatureSupplier(CosineDistanceSignature::new)
      .createVectorStore();

    LabeledVector target = new LabeledVector("TARGET", SparseVector.randomGaussian(100));
    List<LabeledVector> vectors = new ArrayList<>();
    for (int i = 0; i < 50_000; i++) {
      vectors.add(new LabeledVector(StringUtils.randomHexString(10), SparseVector.randomGaussian(100)));
    }
    Stopwatch build = Stopwatch.createStarted();
    vectors.forEach(vectorStore::add);
    build.stop();
    System.out.println("Building: [" + build + "]");
    Stopwatch query = Stopwatch.createStarted();
    vectorStore.nearest(target, 10).forEach(slv -> System.out.println(slv.getLabel() + "\t" + slv.getScore()));
    query.stop();
    System.out.println("Querying: [" + query + "]");
  }


  @Override
  public Set<KEY> keySet() {
    return new HashSet<>(keys.keys());
  }

  @Override
  public boolean containsKey(KEY key) {
    return keySet().contains(key);
  }

  @Override
  protected void removeVector(LabeledVector vector, int id) {
    vectorIDMap.removeKey(id);
    keys.removeKey(vector.getLabel());
  }

  @Override
  public Iterator<LabeledVector> iterator() {
    return Collections.unmodifiableCollection(vectorIDMap.values()).iterator();
  }

  @Override
  protected void registerVector(LabeledVector vector, int id) {
    keys.put(vector.getLabel(), id);
    vectorIDMap.put(id, vector);
  }

  @Override
  protected int nextUniqueID() {
    return vectorIDGenerator.getAndIncrement();
  }

  @Override
  protected int getID(KEY key) {
    return keys.get(key);
  }

  @Override
  protected LabeledVector getVectorByID(int id) {
    return vectorIDMap.get(id);
  }

  @Override
  public int size() {
    return vectorIDMap.size();
  }

}// END OF LSHVectorStore
