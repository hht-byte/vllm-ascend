import unittest


class AudioBoundaryTests(unittest.TestCase):
    def test_vad_boundaries_keep_complete_audio_with_two_to_six_second_chunks(self):
        from blue_prepare_audio import boundaries
        for speech in ([{'start':0,'end':600},{'start':620,'end':1200},{'start':1210,'end':1500}],
                       [{'start':20,'end':410},{'start':440,'end':900},{'start':920,'end':1400}]):
            spans = boundaries(1500,speech,100)
            self.assertEqual(spans[0][0],0)
            self.assertEqual(spans[-1][1],1500)
            self.assertTrue(all(200<=b-a<=600 for a,b in spans))
            self.assertTrue(all(x[1]==y[0] for x,y in zip(spans,spans[1:])))
        self.assertEqual(boundaries(1500,[{'start':0,'end':1500}],100),[(0,600),(600,1200),(1200,1500)])

    def test_no_voice_and_malformed_timestamps_are_rejected(self):
        from blue_prepare_audio import boundaries
        for rows in ([],[{'start':2,'end':1}],[{'start':0,'end':1501}]):
            with self.assertRaises(ValueError):boundaries(1500,rows,100)
