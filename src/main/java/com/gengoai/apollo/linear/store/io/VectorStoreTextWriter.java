package com.gengoai.apollo.linear.store.io;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.json.Json;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonWriter;
import com.gengoai.stream.MStream;
import com.gengoai.string.CharMatcher;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.HashMap;
import java.util.Map;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.notNullOrBlank;

/**
 * @author David B. Bracewell
 */
public class VectorStoreTextWriter implements VectorStoreWriter {
   public static final String INDEX_EXT = ".idx.json.gz";
   private final int dimension;
   private RandomAccessFile vectorWriter;
   private JsonWriter indexWriter;
   private long lastOffset = 0;
   private final File vectorFile;
   private final File indexFile;

   public VectorStoreTextWriter(int dimension, File vectorFile) throws IOException {
      this.dimension = dimension;
      this.vectorFile = vectorFile;
      this.indexFile = new File(vectorFile.getAbsolutePath() + INDEX_EXT);
      this.vectorWriter = new RandomAccessFile(vectorFile, "rw");
      this.indexWriter = new JsonWriter(Resources.fromFile(indexFile).setIsCompressed(true));
      this.indexWriter.beginDocument();
   }

   public static File indexFileFor(File vectorStore) {
      return new File(vectorStore.getAbsolutePath() + INDEX_EXT);
   }

   public static Map<String, Long> readIndexFor(File vectorStore) throws IOException {
      return Json.parse(Resources.fromFile(indexFileFor(vectorStore)))
                 .getAsMap(Long.class);
   }

   public static int determineDimension(File vectorStore) throws IOException {
      Resource r = Resources.fromFile(vectorStore);
      try (MStream<String> stream = r.lines()) {
         return stream.first()
                      .map(line -> {
                         String[] cells = line.trim().split("[ \t]+");
                         if (cells.length > 4) {
                            return cells.length - 1;
                         }
                         return Integer.parseInt(cells[1]);
                      }).orElseThrow(() -> new IllegalStateException("Cannot determine dimension for: " + vectorStore));
      } catch (Exception e) {
         throw new IOException(e);
      }
   }

   public static Map<String, Long> createIndexFor(File vectorStore) throws IOException {
      Map<String, Long> keyOffsets = new HashMap<>();
      try (RandomAccessFile raf = new RandomAccessFile(vectorStore, "r")) {
         String line = raf.readLine();
         long start = raf.getFilePointer();
         String[] cells = line.split("[ \t]+");
         if (cells.length > 4) {
            keyOffsets.put(cells[0], start);
         }
         start = raf.getFilePointer();
         while ((line = raf.readLine()) != null) {
            int i = CharMatcher.WhiteSpace.findIn(line);
            if (i > 0) {
               keyOffsets.put(line.substring(0, i), start);
               start = raf.getFilePointer();
            }
         }
      }
      return keyOffsets;
   }


   public static void writeIndexFor(File vectorStore, Map<String, Long> offsets) throws IOException {
      Json.dump(JsonEntry.from(offsets), Resources.fromFile(indexFileFor(vectorStore)).setIsCompressed(true));
   }

   public File getVectorFile() {
      return vectorFile;
   }

   public File getIndexFile() {
      return indexFile;
   }

   @Override
   public void close() throws IOException {
      indexWriter.endDocument();
      indexWriter.close();
      vectorWriter.close();
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public void write(String key, NDArray vector) throws IOException {
      notNullOrBlank(key, "The key must not be null or blank");
      checkArgument(dimension == vector.length(),
                    () -> "Dimension mismatch. (" + dimension + ") != (" + vector.length() + ")");
      StringBuilder cLine = new StringBuilder(key);
      for (int i = 0; i < vector.length(); i++) {
         cLine.append(" ").append(vector.get(i));
      }
      cLine.append("\n");
      vectorWriter.write(cLine.toString().getBytes());
      indexWriter.property(key, lastOffset);
      lastOffset = vectorWriter.getFilePointer();
   }
}//END OF VectorStoreTextWriter
