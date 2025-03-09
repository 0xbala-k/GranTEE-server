// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract GranTEE {
    uint public scholarshipCounter;

    // Each scholarship is stored with its creator, balance, existence flag,
    // a mapping of fund managers, and an array to track them.
    struct Scholarship {
        address creator;
        uint balance;
        bool exists;
        mapping(address => bool) fundManagers;
        address[] fundManagerList;
    }

    // Because Scholarship contains an internal mapping, we store them in a mapping keyed by an auto-incremented ID.
    mapping(uint => Scholarship) public scholarships;

    // Enum for application status.
    enum ApplicationStatus {
        Pending,
        Reviewed,
        Accepted,
        Rejected
    }

    // Each application now stores if the student applied, if they've been paid, data hash of any details they provided,
    // and the current application status.
    struct Application {
        bool exists;
        bool paid;
        string dataHash;
        ApplicationStatus status;
    }

    // Mapping from scholarship ID to a student's application.
    mapping(uint => mapping(address => Application)) private applications;

    // To allow students to view all the scholarships they've applied to.
    mapping(address => uint[]) private studentApplications;

    // Struct for returning a summary of scholarship data.
    struct ScholarshipView {
        uint scholarshipId;
        address creator;
        uint balance;
    }

    // Struct for returning a student's application details.
    struct ApplicationView {
        uint scholarshipId;
        string dataHash;
        bool paid;
        ApplicationStatus status;
    }

    // Events for logging actions.
    event ScholarshipCreated(uint scholarshipId, address indexed creator);
    event ScholarshipDeleted(
        uint scholarshipId,
        address indexed creator,
        uint refundAmount
    );
    event FundManagerAdded(uint scholarshipId, address indexed manager);
    event FundManagerRemoved(uint scholarshipId, address indexed manager);
    event FundsDeposited(
        uint scholarshipId,
        address indexed depositor,
        uint amount
    );
    event ApplicationSubmitted(
        uint scholarshipId,
        address indexed student,
        string dataHash
    );
    event ScholarshipSent(
        uint scholarshipId,
        address indexed student,
        uint amount
    );
    event ApplicationStatusUpdated(
        uint scholarshipId,
        address indexed student,
        ApplicationStatus status
    );

    // Modifier to ensure the scholarship exists.
    modifier scholarshipExists(uint _scholarshipId) {
        require(
            scholarships[_scholarshipId].exists,
            "Scholarship does not exist"
        );
        _;
    }

    // Modifier to restrict functions to the scholarship creator.
    modifier onlyCreator(uint _scholarshipId) {
        require(
            msg.sender == scholarships[_scholarshipId].creator,
            "Only creator can perform this action"
        );
        _;
    }

    // Modifier to restrict functions to fund managers of the scholarship.
    modifier onlyFundManager(uint _scholarshipId) {
        require(
            scholarships[_scholarshipId].fundManagers[msg.sender],
            "Only fund manager can perform this action"
        );
        _;
    }

    /// @notice Create a new scholarship. The creator is automatically added as a fund manager.
    /// @return The new scholarship ID.
    function createScholarship() external returns (uint) {
        scholarshipCounter++;
        uint scholarshipId = scholarshipCounter;
        Scholarship storage s = scholarships[scholarshipId];
        s.creator = msg.sender;
        s.balance = 0;
        s.exists = true;
        emit ScholarshipCreated(scholarshipId, msg.sender);
        return scholarshipId;
    }

    /// @notice Add a fund manager to a scholarship. Only the scholarship creator can call.
    function addFundManager(
        uint _scholarshipId,
        address _manager
    ) external scholarshipExists(_scholarshipId) onlyCreator(_scholarshipId) {
        require(_manager != address(0), "Invalid address");
        if (!scholarships[_scholarshipId].fundManagers[_manager]) {
            scholarships[_scholarshipId].fundManagers[_manager] = true;
            scholarships[_scholarshipId].fundManagerList.push(_manager);
        }
        emit FundManagerAdded(_scholarshipId, _manager);
    }

    /// @notice Remove a fund manager from a scholarship. Only the scholarship creator can call.
    function removeFundManager(
        uint _scholarshipId,
        address _manager
    ) external scholarshipExists(_scholarshipId) onlyCreator(_scholarshipId) {
        require(
            scholarships[_scholarshipId].fundManagers[_manager],
            "Address is not a fund manager"
        );
        scholarships[_scholarshipId].fundManagers[_manager] = false;
        // Note: The address remains in fundManagerList but is filtered out in getFundManagers.
        emit FundManagerRemoved(_scholarshipId, _manager);
    }

    /// @notice Deposit funds into a specific scholarship. Only the creator can deposit.
    function depositFunds(
        uint _scholarshipId
    )
        external
        payable
        scholarshipExists(_scholarshipId)
        onlyCreator(_scholarshipId)
    {
        require(msg.value > 0, "Deposit must be greater than zero");
        scholarships[_scholarshipId].balance += msg.value;
        emit FundsDeposited(_scholarshipId, msg.sender, msg.value);
    }

    /// @notice Delete a scholarship. Only the creator can call. Refunds any remaining funds to the creator.
    function deleteScholarship(
        uint _scholarshipId
    ) external scholarshipExists(_scholarshipId) onlyCreator(_scholarshipId) {
        uint refundAmount = scholarships[_scholarshipId].balance;
        // Mark the scholarship as non-existent.
        scholarships[_scholarshipId].exists = false;
        // Refund remaining funds to the creator.
        if (refundAmount > 0) {
            scholarships[_scholarshipId].balance = 0;
            payable(msg.sender).transfer(refundAmount);
        }
        emit ScholarshipDeleted(_scholarshipId, msg.sender, refundAmount);
    }

    /// @notice Apply to a specific scholarship with provided data hash of the details.
    function applyScholarship(
        uint _scholarshipId,
        string calldata _dataHash
    ) external scholarshipExists(_scholarshipId) {
        Application storage app = applications[_scholarshipId][msg.sender];
        require(!app.exists, "Already applied to this scholarship");
        app.exists = true;
        app.paid = false;
        app.dataHash = _dataHash;
        app.status = ApplicationStatus.Pending;
        studentApplications[msg.sender].push(_scholarshipId);
        emit ApplicationSubmitted(_scholarshipId, msg.sender, _dataHash);
    }

    /// @notice Send scholarship funds from a specific scholarship to an applied student.
    /// Only fund managers for that scholarship can call this function.
    function sendScholarship(
        uint _scholarshipId,
        address payable _student,
        uint _amount
    )
        external
        scholarshipExists(_scholarshipId)
        onlyFundManager(_scholarshipId)
    {
        Application storage app = applications[_scholarshipId][_student];
        require(app.exists, "Student has not applied to this scholarship");
        require(!app.paid, "Scholarship already sent to this student");
        require(
            scholarships[_scholarshipId].balance >= _amount,
            "Insufficient funds in scholarship"
        );

        app.paid = true;
        app.status = ApplicationStatus.Accepted;
        scholarships[_scholarshipId].balance -= _amount;
        _student.transfer(_amount);
        emit ScholarshipSent(_scholarshipId, _student, _amount);
    }

    /// @notice Update the status of a student's application.
    /// Only fund managers of the scholarship can update the application status.
    function updateApplicationStatus(
        uint _scholarshipId,
        address _student,
        ApplicationStatus _newStatus
    )
        external
        scholarshipExists(_scholarshipId)
        onlyFundManager(_scholarshipId)
    {
        Application storage app = applications[_scholarshipId][_student];
        require(app.exists, "Application does not exist");
        app.status = _newStatus;
        emit ApplicationStatusUpdated(_scholarshipId, _student, _newStatus);
    }

    /// @notice Get all applications submitted by the caller.
    /// @return An array of ApplicationView structs containing the scholarship ID, details, payment status, and status.
    function getUserApplications()
        external
        view
        returns (ApplicationView[] memory)
    {
        uint[] storage appliedScholarships = studentApplications[msg.sender];
        ApplicationView[] memory result = new ApplicationView[](
            appliedScholarships.length
        );
        for (uint i = 0; i < appliedScholarships.length; i++) {
            uint schId = appliedScholarships[i];
            Application storage app = applications[schId][msg.sender];
            result[i] = ApplicationView({
                scholarshipId: schId,
                dataHash: app.dataHash,
                paid: app.paid,
                status: app.status
            });
        }
        return result;
    }

    /// @notice Get a list of all existing scholarships.
    /// @return An array of ScholarshipView structs for all scholarships that exist.
    function getActiveScholarships()
        external
        view
        returns (ScholarshipView[] memory)
    {
        uint count = 0;
        for (uint i = 1; i <= scholarshipCounter; i++) {
            if (scholarships[i].exists) {
                count++;
            }
        }
        ScholarshipView[] memory list = new ScholarshipView[](count);
        uint index = 0;
        for (uint i = 1; i <= scholarshipCounter; i++) {
            if (scholarships[i].exists) {
                list[index] = ScholarshipView(
                    i,
                    scholarships[i].creator,
                    scholarships[i].balance
                );
                index++;
            }
        }
        return list;
    }

    /// @notice Get a list of all existing scholarships created by a specific creator.
    /// @param _creator The address of the scholarship creator.
    /// @return An array of ScholarshipView structs for scholarships created by _creator.
    function getScholarshipsByCreator(
        address _creator
    ) external view returns (ScholarshipView[] memory) {
        uint count = 0;
        for (uint i = 1; i <= scholarshipCounter; i++) {
            if (scholarships[i].exists && scholarships[i].creator == _creator) {
                count++;
            }
        }
        ScholarshipView[] memory list = new ScholarshipView[](count);
        uint index = 0;
        for (uint i = 1; i <= scholarshipCounter; i++) {
            if (scholarships[i].exists && scholarships[i].creator == _creator) {
                list[index] = ScholarshipView(
                    i,
                    scholarships[i].creator,
                    scholarships[i].balance
                );
                index++;
            }
        }
        return list;
    }

    /// @notice Get the list of current fund managers for a given scholarship.
    /// @param _scholarshipId The scholarship ID.
    /// @return An array of addresses that are currently active fund managers.
    function getFundManagers(
        uint _scholarshipId
    )
        external
        view
        scholarshipExists(_scholarshipId)
        returns (address[] memory)
    {
        Scholarship storage s = scholarships[_scholarshipId];
        uint activeCount = 0;
        // Count the active fund managers.
        for (uint i = 0; i < s.fundManagerList.length; i++) {
            if (s.fundManagers[s.fundManagerList[i]]) {
                activeCount++;
            }
        }
        address[] memory activeManagers = new address[](activeCount);
        uint j = 0;
        // Collect only the active fund managers.
        for (uint i = 0; i < s.fundManagerList.length; i++) {
            if (s.fundManagers[s.fundManagerList[i]]) {
                activeManagers[j] = s.fundManagerList[i];
                j++;
            }
        }
        return activeManagers;
    }

    // Fallback functions to receive Ether in case funds are sent directly.
    receive() external payable {}

    fallback() external payable {}
}
