# https://huggingface.co/datasets/maveriq/bigbenchhard
bigbenchhard = [
    {
        "input": "not not False and not not not False is",
        "target": "False"
    }
]

# https://huggingface.co/datasets/Idavidrein/gpqa
gpqa = [
    {
        "Question": "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?",
        "Incorrect Answer 1": "10^-11 eV",
        "Incorrect Answer 2": "10^-8 eV",
        "Incorrect Answer 3": "10^-9 eV",
        "Correct Answer": "10^-4 eV"
    }
]

# https://huggingface.co/datasets/openai/gsm8k
gsm8k = [
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"
    }
]

# https://huggingface.co/datasets/greengerong/leetcode
leetcode = [
    {
        "title": "Median of Two Sorted Arrays",
        "difficulty": "Hard",
        "content": """
            Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return **the median** of the two sorted arrays.

            The overall run time complexity should be `O(log (m+n))`.

            **Example 1:**

            **Input:** nums1 = \[1,3\], nums2 = \[2\]
            **Output:** 2.00000
            **Explanation:** merged array = \[1,2,3\] and median is 2.

            **Example 2:**

            **Input:** nums1 = \[1,2\], nums2 = \[3,4\]
            **Output:** 2.50000
            **Explanation:** merged array = \[1,2,3,4\] and median is (2 + 3) / 2 = 2.5.

            **Constraints:**

            * `nums1.length == m`
            * `nums2.length == n`
            * `0 <= m <= 1000`
            * `0 <= n <= 1000`
            * `1 <= m + n <= 2000`
            * `-106 <= nums1[i], nums2[i] <= 106`
        """,
        "python": """"
            ```python
            def findMedianSortedArrays(nums1, nums2):
            if len(nums1) > len(nums2):
            return findMedianSortedArrays(nums2, nums1)

            x, y = len(nums1), len(nums2)
            low, high = 0, x

            while low <= high:
            partition_x = (low + high) // 2
            partition_y = (x + y + 1) // 2 - partition_x

            max_left_x = float('-inf') if partition_x == 0 else nums1[partition_x - 1]
            min_right_x = float('inf') if partition_x == x else nums1[partition_x]

            max_left_y = float('-inf') if partition_y == 0 else nums2[partition_y - 1]
            min_right_y = float('inf') if partition_y == y else nums2[partition_y]

            if max_left_x <= min_right_y and max_left_y <= min_right_x:
            if (x + y) % 2 == 0:
            return (max(max_left_x, max_left_y) + min(min_right_x, min_right_y)) / 2
            else:
            return max(max_left_x, max_left_y)
            elif max_left_x > min_right_y:
            high = partition_x - 1
            else:
            low = partition_x + 1

            return 0
            ```

            1. Choose the smaller array as nums1 so that the problem is simpler with less log(n) complexity.
            2. Use Binary Search (BS) to partition the smallest array.
            3. Now we calculate the position of partition in the larger array (nums2) having fetched the smaller one.
            4. Find the four important numbers - maxSize - left and right of partition in the two arrays.
            5. If maxSizeLeft <= minSizeRight and maxSizeLeft2 <= minSizeRight2, then the partition of both arrays is correct, if not, adjust the partition of nums1. If maxLeftX > minRightY, move the BS partition to the left; if maxLeftY > minRightX, move the BS partition to the right.
            6. When the correct partition is found, calculate the median based on the length of the merged array, even or odd.
        """
    }
]

# https://huggingface.co/datasets/cais/mmlu
mmlu = [
    {
        "question": "The polynomial x^3 + 2x^2 + 2x + 1 can be factored into linear factors in Z_7[x]. Find this factorization.",
        "choices": [
            "(x - 2)(x + 2)(x - 1)",
            "(x + 1)(x + 4)(x - 2)",
            "(x + 1)(x - 4)(x - 2)",
            "(x - 1)(x - 4)(x - 2)"
        ],
        "answer": "2 C"
    }
]

def get_sample_datasets(dataset):
    match dataset:
        case "bigbenchhard":
            return bigbenchhard
        case "gpqa":
            return gpqa
        case "gsm8k":
            return gsm8k
        case "leetcode":
            return leetcode
        case "mmlu":
            return mmlu
    return [{}]