package com.gengoai.apollo.linear.store;

import com.gengoai.Parameters;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.store.io.VectorStoreTextWriter;
import com.gengoai.cache.AutoCalculatingLRUCache;
import com.gengoai.cache.Cache;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.logging.Loggable;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.Serializable;
import java.util.*;

import static com.gengoai.Parameters.params;
import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.notNullOrBlank;

/**
 * The type Indexed file store.
 *
 * @author David B. Bracewell
 */
public final class DiskBasedVectorStore implements VectorStore, Serializable, Loggable {
   private static final long serialVersionUID = 1L;
   private final Map<String, Long> keyOffsets;
   private final File vectorFile;
   private transient final Cache<String, NDArray> vectorCache;
   private final int dimension;
   private final int cacheSize;

   public DiskBasedVectorStore(File vectorFile, int cacheSize) {
      try {
         this.vectorFile = vectorFile;
         this.keyOffsets = new HashMap<>();
         this.cacheSize = cacheSize;
         this.dimension = VectorStoreTextWriter.determineDimension(vectorFile);
         File indexFile = VectorStoreTextWriter.indexFileFor(vectorFile);
         if (indexFile.exists()) {
            keyOffsets.putAll(VectorStoreTextWriter.readIndexFor(vectorFile));
         } else {
            keyOffsets.putAll(VectorStoreTextWriter.createIndexFor(vectorFile));
            VectorStoreTextWriter.writeIndexFor(vectorFile, keyOffsets);
         }
         vectorCache = new AutoCalculatingLRUCache<>(cacheSize, this::loadNDArray);
      } catch (Exception e) {
         throw new RuntimeException(e);
      }
   }

   @Override
   public void write(Resource location) throws IOException {
      Resource vectors = Resources.fromFile(vectorFile);
      Resource index = Resources.fromFile(VectorStoreTextWriter.indexFileFor(vectorFile));
      vectors.copy(location);
      index.copy(Resources.fromFile(VectorStoreTextWriter.indexFileFor(location.asFile()
                                                                               .orElseThrow(IOException::new))));
   }

   /**
    * Builder builder.
    *
    * @return the builder
    */
   public static Builder builder() {
      return new Builder();
   }

   @Override
   public boolean containsKey(String s) {
      return keyOffsets.containsKey(s);
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public NDArray get(String s) {
      return vectorCache.get(s);
   }

   @Override
   public Iterator<NDArray> iterator() {
      return new Iterator<NDArray>() {
         private final Iterator<String> itr = keyOffsets.keySet().iterator();

         @Override
         public boolean hasNext() {
            return itr.hasNext();
         }

         @Override
         public NDArray next() {
            return vectorCache.get(itr.next());
         }
      };
   }

   @Override
   public Set<String> keySet() {
      return Collections.unmodifiableSet(keyOffsets.keySet());
   }

   private NDArray loadNDArray(String key) {
      if (keyOffsets.containsKey(key)) {
         try (RandomAccessFile raf = new RandomAccessFile(vectorFile, "r")) {
            raf.seek(keyOffsets.get(key));
            return VectorStoreTextWriter.lineToVector(raf.readLine(), dimension);
         } catch (Exception e) {
            throw new RuntimeException(e);
         }
      }
      return NDArrayFactory.SPARSE.zeros(dimension);
   }

   @Override
   public int size() {
      return keyOffsets.size();
   }

   @Override
   public Parameters<VSParams> getParameters() {
      return params(VSParams.IN_MEMORY, false,
                    VSParams.CACHE_SIZE, cacheSize,
                    VSParams.LOCATION, vectorFile.getAbsolutePath());
   }

   /**
    * The type Builder.
    */
   public static class Builder implements VSBuilder {
      private VectorStoreTextWriter writer = null;
      private int dimension = -1;

      private void ensureWriter() {
         if (writer == null) {
            try {
               writer = new VectorStoreTextWriter(dimension,
                                                  Resources.temporaryFile()
                                                           .asFile()
                                                           .orElseThrow(IllegalStateException::new));
            } catch (IOException e) {
               throw new RuntimeException(e);
            }
         }
      }

      @Override
      public VSBuilder add(String key, NDArray vector) {
         notNullOrBlank(key, "The key must not be null or blank");
         try {
            if (dimension == -1) {
               dimension = (int) vector.length();
            }
            checkArgument(dimension == vector.length(),
                          () -> "Dimension mismatch. (" + dimension + ") != (" + vector.length() + ")");
            ensureWriter();
            writer.write(key, vector);
         } catch (IOException e) {
            throw new RuntimeException(e);
         }
         return this;
      }

      @Override
      public VectorStore build(Parameters<VSParams> params) throws IOException {
         File location = new File(params.getString(VSParams.LOCATION));
         File indexLocation = VectorStoreTextWriter.indexFileFor(location);
         if (writer != null) {
            writer.close();
            Resource vectors = Resources.fromFile(writer.getVectorFile());
            Resource index = Resources.fromFile(writer.getIndexFile());
            vectors.copy(Resources.fromFile(location));
            index.copy(Resources.fromFile(indexLocation));
         }
         writer = null;
         return new DiskBasedVectorStore(location, params.getInt(VSParams.CACHE_SIZE));
      }

   }

}//END OF DiskBasedVectorStore
